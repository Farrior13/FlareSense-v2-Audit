"""Step 01 - run the published FlareSense-v2 checkpoint on an HF split.

The checkpoint is `i4ds/flaresense-v2/model.ckpt` (uploaded 2025-01-07, config
best_v2.yml). Preprocessing replicates the original repository exactly
(ecallisto_dataset.EcallistoDataset.__getitem__ with best_v2.yml, custom_resize=False):

    ToTensor (uint8 -> [0,1])  ->  subtract per-frequency-row median  ->
    torchvision Resize((128, 512))  ->  per-image min-max scaling to [0,1]  ->
    grayscale expanded to 3 channels  ->  ResNet-34 (1 output logit)

Output: data/flaresense_v2_predictions_{split}.parquet with columns
start_datetime, antenna, manual_label, v2_logit (row order = HF split order).

Usage:  python analysis/01_run_v2_inference.py --split test   (GPU: ~1-2 min; CPU: ~30-60 min)
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import Resize, ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HF_DATASET, V2_PRED_FILE  # noqa: E402

_to_tensor, _resize = ToTensor(), Resize((128, 512))


def preprocess(img) -> torch.Tensor:
    img = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else img
    x = _to_tensor(img)
    if x.dim() == 3:
        x = x.squeeze(0)
    x = x - torch.median(x, dim=1).values[:, None]   # remove_background
    x = _resize(x.unsqueeze(0)).squeeze(0)           # normal_resize
    x = (x - x.min()) / (x.max() - x.min())          # min_max_scaler
    return x.unsqueeze(0)


class _DS(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return preprocess(self.d[i]["image"]), i


def load_flaresense_v2(device):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("i4ds/flaresense-v2", "model.ckpt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k.replace("resnet._orig_mod.", "", 1): v for k, v in sd.items() if k.startswith("resnet._orig_mod.")}
    net = models.resnet34(weights=None, num_classes=1)
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval(), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from datasets import load_dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, path = load_flaresense_v2(device)
    print(f"checkpoint: {path}  device: {device}", flush=True)

    ds = load_dataset(HF_DATASET, split=a.split)
    meta = ds.select_columns(["start_datetime", "antenna", "manual_label"]).to_pandas()
    dl = DataLoader(_DS(ds), batch_size=a.batch_size, num_workers=a.workers, shuffle=False)
    logits = np.full(len(ds), np.nan, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b, (x, idx) in enumerate(dl):
            out = net(x.to(device, non_blocking=True).expand(-1, 3, -1, -1)).squeeze(1)
            logits[idx.numpy()] = out.float().cpu().numpy()
            if b % 50 == 0:
                print(f"  {b * a.batch_size:>7}/{len(ds)}  {time.time() - t0:6.0f}s", flush=True)
    assert not np.isnan(logits).any()
    meta["v2_logit"] = logits
    out = Path(str(V2_PRED_FILE).format(split=a.split))
    meta.to_parquet(out, index=False)
    print(f"saved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
