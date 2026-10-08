"""Controlled retraining experiment: does removing event overlap with the test set change
what FlareSense-v2-style training achieves on the ORIGINAL test set?

Design (all arms share the same pipeline, hyper-parameters and calibration set; only the
training pool differs):

  arm = full            train on HF train+val (minus calibration set)          -> pipeline-fidelity check
  arm = purged          ... minus every sample within +-PURGE minutes of ANY test burst
                        (any station)  -> no physical event is shared with the test set
  arm = random_control  ... minus the SAME number of positives/negatives as `purged`,
                        drawn at random  -> controls for the smaller training set

Evaluation is always on the untouched HF `test` split with its PI re-verified labels, i.e.
exactly the set and labels behind the paper's Table 3. Comparing purged vs random_control
on (i) leaked vs clean test bursts and (ii) catalog-listed vs PI-added bursts identifies
the causal effect of event overlap, holding label protocol and data volume fixed.

Training replicates configs/best_v2.yml + main.py of i4Ds/FlareSense-v2:
  ResNet-34 (from scratch, 1 logit, grayscale->3ch), input 128x512, batch 64, AdamW
  lr 2.3763e-4, wd 5.165e-4, 25 epochs, per-epoch LR schedule: linear warm-up from 0.1x over
  12 epochs, then linear decay to 0 over 13 epochs, BCE-with-logits with label smoothing
  0.1174 (the original passes a *scalar* class weight to `weight=`, which only rescales the
  loss uniformly -> no effect under Adam; hence plain BCE here, NO pos_weight),
  AMP fp16, augmentation: TimeWarp(W=389) before resize + SpecAugment(f=25, t=70, 'random')
  after resize, final-epoch model (no checkpoint selection, as in the original).

Usage:
  python train/train_leakage_experiment.py --arm purged --seed 0
  python train/train_leakage_experiment.py --arm random_control --seed 0
  python train/train_leakage_experiment.py --arm purged --seed 0 --smoke   (2 tiny epochs)
Resumable: rerun the same command; it continues from the last completed epoch.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import Resize, ToTensor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "retrain"
CKPT = ROOT / "checkpoints"
HF = "i4ds/ecallisto_radio_sunburst"

CFG = dict(batch_size=64, epochs=25, warmup_epochs=12, lr=0.00023762695665743765,
           weight_decay=0.0005164989055101516, label_smoothing=0.11739565855800864,
           input_size=(128, 512), time_warp_w=389, freq_mask=25, time_mask=70)

_to_tensor = ToTensor()
_resize = Resize(CFG["input_size"])


# ----------------------------------------------------------------- augmentation (verbatim logic of the original)
class TimeWarp:
    """TimeWarpAugmenter from ecallisto_dataset.py (spline warp along time axis)."""
    def __init__(self, W): self.W = W

    @staticmethod
    def _h_poly(t):
        tt = t.unsqueeze(-2) ** torch.arange(4, device=t.device).view(-1, 1)
        A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
        return A @ tt

    @classmethod
    def _interp(cls, x, y, xs):
        m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
        m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], -1)
        idxs = torch.searchsorted(x[..., 1:], xs)
        dx = x.gather(-1, idxs + 1) - x.gather(-1, idxs)
        hh = cls._h_poly((xs - x.gather(-1, idxs)) / dx)
        return (hh[..., 0, :] * y.gather(-1, idxs) + hh[..., 1, :] * m.gather(-1, idxs) * dx
                + hh[..., 2, :] * y.gather(-1, idxs + 1) + hh[..., 3, :] * m.gather(-1, idxs + 1) * dx)

    def __call__(self, spec):  # spec: (H, W_time)
        W = self.W
        if spec.size(-1) <= 2 * W:
            return spec
        specs = spec.unsqueeze(0).unsqueeze(0)
        b, _, rows, L = specs.shape
        p = torch.randint(W, L - W, (b,))
        d = torch.randint(-W, W, (b,))
        x = torch.stack([torch.zeros(b), p.float(), torch.full((b,), L - 1.0)], 1)
        y = torch.stack([torch.full((b,), -1.0), (p - d).float() * 2 / (L - 1.0) - 1.0, torch.ones(b)], 1)
        xs = torch.linspace(0, L - 1, L).unsqueeze(0).expand(b, -1)
        ys = self._interp(x, y, xs)
        grid = torch.cat((ys.view(b, 1, -1, 1).expand(-1, rows, -1, -1),
                          torch.linspace(-1, 1, rows).view(-1, 1, 1).expand(b, -1, L, -1)), -1)
        return F.grid_sample(specs, grid, align_corners=True).squeeze(0).squeeze(0)


def spec_augment(x, f_par, t_par):
    """CustomSpecAugment(method='random'): one freq band and one time band set to U(0,1)."""
    if f_par > 0:
        pad = torch.rand(1)
        m = int(torch.randint(0, f_par + 1, (1,)))
        f = int(torch.randint(0, x.size(0) - m + 1, (1,)))
        x[f:f + m, :] = pad
    if t_par > 0:
        pad = torch.rand(1)
        m = int(torch.randint(0, t_par + 1, (1,)))
        t = int(torch.randint(0, x.size(1) - m + 1, (1,)))
        x[:, t:t + m] = pad
    return x


def load_image(img_field):
    img = Image.open(io.BytesIO(img_field["bytes"])) if isinstance(img_field, dict) else img_field
    x = _to_tensor(img)
    return x.squeeze(0) if x.dim() == 3 else x


class SpecDataset(Dataset):
    def __init__(self, hf: dict, rows: pd.DataFrame, train: bool):
        self.hf, self.train = hf, train
        self.split = rows.hf_split.values
        self.idx = rows.hf_index.values.astype(int)
        self.y = rows.y.values.astype(np.float32)
        self.tw = TimeWarp(CFG["time_warp_w"])

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = load_image(self.hf[self.split[i]][int(self.idx[i])]["image"])
        x = x - torch.median(x, dim=1).values[:, None]                 # remove_background
        if self.train:
            x = self.tw(x)                                             # augm_before_resize
        x = _resize(x.unsqueeze(0)).squeeze(0)                         # resize
        if self.train:
            x = spec_augment(x, CFG["freq_mask"], CFG["time_mask"])    # augm_after_resize
        rng = x.max() - x.min()
        x = (x - x.min()) / rng if rng > 0 else torch.zeros_like(x)    # min_max_scaler
        return x.unsqueeze(0), torch.tensor([self.y[i]]), i


# ----------------------------------------------------------------- pools
def build_pools(seed: int, purge_min: int, calib_frac: float):
    def meta(split):
        f = ROOT / "data" / f"meta_{split}.parquet"
        d = pd.read_parquet(f, columns=["manual_label", "start_datetime", "antenna"])
        d["hf_split"], d["hf_index"] = split, np.arange(len(d))
        return d
    tv = pd.concat([meta("train"), meta("val")], ignore_index=True)
    te = meta("test")
    for d in (tv, te):
        d["start_datetime"] = pd.to_datetime(d.start_datetime)
        d["y"] = (d.manual_label != 0).astype(int)

    # purge flag: within +-purge_min of any TEST burst start (any station)
    tb = np.sort(te[te.y == 1].start_datetime.values.astype("datetime64[s]").astype(np.int64))
    t = tv.start_datetime.values.astype("datetime64[s]").astype(np.int64)
    lo = np.searchsorted(tb, t - purge_min * 60, "left")
    hi = np.searchsorted(tb, t + purge_min * 60, "right")
    tv["purge"] = hi > lo

    rng = np.random.default_rng(1000 + seed)
    # calibration set: stratified random among NON-purged samples (identical for all arms of a seed)
    calib = np.zeros(len(tv), bool)
    for cls in (0, 1):
        cand = np.flatnonzero((tv.y.values == cls) & ~tv.purge.values)
        calib[rng.choice(cand, int(round(calib_frac * (tv.y.values == cls).sum())), replace=False)] = True
    tv["calib"] = calib
    return tv, te


def select_arm(tv: pd.DataFrame, arm: str, seed: int) -> pd.DataFrame:
    pool = tv[~tv.calib]
    if arm == "full":
        return pool
    if arm == "purged":
        return pool[~pool.purge]
    if arm == "random_control":
        rng = np.random.default_rng(2000 + seed)
        drop = []
        for cls in (0, 1):
            n = int((pool.purge & (pool.y == cls)).sum())
            cand = pool.index[pool.y == cls].values
            drop.append(rng.choice(cand, n, replace=False))
        return pool.drop(index=np.concatenate(drop))
    raise ValueError(arm)


# ----------------------------------------------------------------- train / predict
def lr_factor(epoch0: int, warm: int, total: int) -> float:
    """Per-epoch factor reproducing SequentialLR(LinearLR(0.1->1, warm), LinearLR(1->0, total-warm))."""
    if epoch0 < warm:
        return 0.1 + 0.9 * epoch0 / warm
    return max(0.0, 1.0 - (epoch0 - warm) / (total - warm))


@torch.no_grad()
def predict(net, loader, device):
    net.eval()
    out = np.full(len(loader.dataset), np.nan, np.float32)
    for x, _, i in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            z = net(x.to(device, non_blocking=True).expand(-1, 3, -1, -1)).squeeze(1)
        out[i.numpy()] = z.float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full", "purged", "random_control"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--purge-minutes", type=int, default=30)
    ap.add_argument("--calib-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    tag = f"{a.arm}_seed{a.seed}" + ("_smoke" if a.smoke else "")
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(exist_ok=True)
    ck_path = CKPT / f"{tag}.pt"
    log_path = OUT / f"{tag}_history.json"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    tv, te = build_pools(a.seed, a.purge_minutes, a.calib_frac)
    train_rows = select_arm(tv, a.arm, a.seed)
    calib_rows = tv[tv.calib]
    epochs, warm = a.epochs, CFG["warmup_epochs"]
    if a.smoke:
        train_rows = train_rows.sample(512, random_state=0); calib_rows = calib_rows.sample(256, random_state=0)
        te = te.sample(256, random_state=0); epochs, warm = 2, 1
    info = {"arm": a.arm, "seed": a.seed, "purge_minutes": a.purge_minutes, "n_train": len(train_rows),
            "n_train_pos": int(train_rows.y.sum()), "n_calib": len(calib_rows), "n_calib_pos": int(calib_rows.y.sum()),
            "n_purged_total": int(tv.purge.sum()), "n_purged_pos": int((tv.purge & (tv.y == 1)).sum()),
            "n_test": len(te), "cfg": CFG}
    print(json.dumps(info, indent=1), flush=True)

    from datasets import load_dataset
    hf = {s: load_dataset(HF, split=s) for s in ("train", "val", "test")}
    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    train_dl = DataLoader(SpecDataset(hf, train_rows, True), batch_size=CFG["batch_size"], shuffle=True,
                          drop_last=False, **dl_kw)
    calib_dl = DataLoader(SpecDataset(hf, calib_rows, False), batch_size=128, shuffle=False, **dl_kw)

    net = models.resnet34(weights=None, num_classes=1).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    start, history = 0, []
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start, history = ck["epoch_done"], ck["history"]
        if "rng_cpu" in ck:
            torch.set_rng_state(ck["rng_cpu"].cpu())
        print(f"resumed after epoch {start}", flush=True)

    ls = CFG["label_smoothing"]
    for ep in range(start, epochs):
        for g in opt.param_groups:
            g["lr"] = CFG["lr"] * lr_factor(ep, warm, epochs)
        net.train(); t0 = time.time(); tot, n = 0.0, 0
        for b, (x, y, _) in enumerate(train_dl):
            x = x.to(device, non_blocking=True).expand(-1, 3, -1, -1); y = y.to(device, non_blocking=True)
            y_s = y * (1 - ls) + (1 - y) * ls
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = F.binary_cross_entropy_with_logits(net(x), y_s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
            if b % 200 == 0:
                print(f"ep {ep+1}/{epochs} b {b}/{len(train_dl)} loss {tot/n:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{n/(time.time()-t0):.0f} smp/s", flush=True)
        # monitoring only (no model selection): calibration-set loss / AP
        z = predict(net, calib_dl, device); yc = calib_rows.y.values
        from sklearn.metrics import average_precision_score, roc_auc_score
        p = 1 / (1 + np.exp(-z))
        rec = {"epoch": ep + 1, "train_loss": tot / n, "lr": opt.param_groups[0]["lr"],
               "calib_ap": float(average_precision_score(yc, p)), "calib_auc": float(roc_auc_score(yc, p)),
               "calib_bce": float(F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(yc, dtype=torch.float32))),
               "sec": time.time() - t0}
        history.append(rec); print(json.dumps(rec), flush=True)
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch_done": ep + 1, "history": history, "rng_cpu": torch.get_rng_state(), "info": info}, ck_path)
        json.dump({"info": info, "history": history}, open(log_path, "w"), indent=1)

    # final-epoch model -> predictions on calibration set and the untouched HF test split
    test_dl = DataLoader(SpecDataset(hf, te, False), batch_size=128, shuffle=False, **dl_kw)
    zt = predict(net, test_dl, device); zc = predict(net, calib_dl, device)
    te.assign(logit=zt)[["hf_index", "start_datetime", "antenna", "y", "logit"]].to_parquet(OUT / f"{tag}_test.parquet", index=False)
    calib_rows.assign(logit=zc)[["hf_split", "hf_index", "start_datetime", "antenna", "y", "logit"]].to_parquet(
        OUT / f"{tag}_calib.parquet", index=False)
    torch.save(net.state_dict(), CKPT / f"{tag}_final_weights.pt")
    print(f"done: {tag}", flush=True)


if __name__ == "__main__":
    main()
