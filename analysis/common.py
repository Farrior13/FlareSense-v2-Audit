"""Shared utilities for the FlareSense-v2 event-leakage analysis.

Everything here operates on *metadata only* (labels, timestamps, antenna,
model outputs). Spectrogram images are never decoded except in
`01_run_v2_inference.py`.

Conventions
-----------
* burst label:            y = (manual_label != 0)
* FlareSense-v2 output:   `v2_logit` (raw logit of the published checkpoint)
* paper calibration:      p = sigmoid(v2_logit / T), T = 0.4974 (pred_live.py)
* paper threshold:        p >= 0.426 (paper Sec. 6, matched-precision operating point)
* "old" predictions:      HF columns `prob` / `model_label` (earlier ResNet, uploaded 2024-10-19;
                          NOT the FlareSense-v2 checkpoint; kept only as a reference model)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (DATA, RESULTS, FIGURES):
    _d.mkdir(exist_ok=True)

HF_DATASET = "i4ds/ecallisto_radio_sunburst"
T_CAL = 0.4974
THR_CAL = 0.426
META_COLS = ["manual_label", "logits", "prob", "model_label", "start_datetime", "antenna"]
V2_PRED_FILE = DATA / "flaresense_v2_predictions_{split}.parquet"


# ----------------------------------------------------------------------------- data
def load_meta(split: str) -> pd.DataFrame:
    """Metadata of one HF split (no images). Cached to data/meta_{split}.parquet."""
    cache = DATA / f"meta_{split}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET, split=split)
        df = ds.select_columns(META_COLS).to_pandas()
        df.to_parquet(cache)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    df["y"] = (df["manual_label"] != 0).astype(int)
    return df


def load_split_with_v2(split: str = "test") -> pd.DataFrame:
    """HF metadata + FlareSense-v2 logits (row order identical to the HF split)."""
    df = load_meta(split)
    f = Path(str(V2_PRED_FILE).format(split=split))
    if not f.exists():
        raise FileNotFoundError(f"{f} missing - run analysis/01_run_v2_inference.py --split {split}")
    v2 = pd.read_parquet(f)
    assert len(v2) == len(df), "row count mismatch between HF split and v2 predictions"
    # integrity: keys must match row by row
    assert (pd.to_datetime(v2["start_datetime"]).values == df["start_datetime"].values).all()
    assert (v2["antenna"].values == df["antenna"].values).all()
    df["v2_logit"] = v2["v2_logit"].values
    df["v2_prob"] = sigmoid(df["v2_logit"].values / T_CAL)
    df["v2_pred"] = (df["v2_prob"] >= THR_CAL).astype(int)
    df["v2_pred_05"] = (df["v2_logit"] > 0).astype(int)
    df["old_prob"] = df["prob"].values
    df["old_pred"] = df["model_label"].astype(int).values
    return df


def load_trainval() -> pd.DataFrame:
    return pd.concat([load_meta("train"), load_meta("val")], ignore_index=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


MODELS = {  # name -> (pred column, prob column)
    "FlareSense-v2": ("v2_pred", "v2_prob"),
    "old_HF_predictions": ("old_pred", "old_prob"),
}


# ----------------------------------------------------------------------------- event overlap
def bucket(ts: pd.Series, freq: str) -> pd.Series:
    return ts.dt.floor(freq)


def leak_flags(test: pd.DataFrame, tv: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean arrays (len(test)) marking test *burst* samples whose event overlaps a
    train/val burst under several definitions. Non-burst samples are always False."""
    tvb = tv[tv.y == 1]
    is_b = test.y.values == 1
    out = {}
    for f in ("15min", "1h"):
        shared = set(bucket(tvb.start_datetime, f))
        out[f"bucket_{f}"] = is_b & bucket(test.start_datetime, f).isin(shared).values
    tv_t = np.sort(tvb.start_datetime.values.astype("datetime64[s]").astype(np.int64))
    tt = test.start_datetime.values.astype("datetime64[s]").astype(np.int64)
    for w in (15, 30):
        lo = np.searchsorted(tv_t, tt - w * 60, "left")
        hi = np.searchsorted(tv_t, tt + w * 60, "right")
        out[f"rolling_pm{w}min"] = is_b & (hi > lo)
    return out


def event_context(test: pd.DataFrame, tv: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Per test row: number of distinct stations with a burst in the same bucket
    (all splits / train+val only / test only)."""
    tvb = tv[tv.y == 1][["start_datetime", "antenna"]].assign(src="tv")
    teb = test[test.y == 1][["start_datetime", "antenna"]].assign(src="test")
    allb = pd.concat([tvb, teb])
    allb["b"] = bucket(allb.start_datetime, freq)
    n_all = allb.groupby("b").antenna.nunique()
    n_tv = allb[allb.src == "tv"].groupby("b").antenna.nunique()
    n_te = allb[allb.src == "test"].groupby("b").antenna.nunique()
    b = bucket(test.start_datetime, freq)
    return pd.DataFrame({
        "event_bucket": b.values,
        "n_st_all": b.map(n_all).fillna(0).astype(int).values,
        "n_st_tv": b.map(n_tv).fillna(0).astype(int).values,
        "n_st_test": b.map(n_te).fillna(0).astype(int).values,
    }, index=test.index)


# ----------------------------------------------------------------------------- metrics
def confusion(y, p):
    y = np.asarray(y).astype(bool); p = np.asarray(p).astype(bool)
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    return tp, fp, fn, tn


def prf(y, p) -> dict:
    tp, fp, fn, tn = confusion(y, p)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    F = 2 * P * R / (P + R) if (P + R) else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return dict(precision=P, recall=R, f1=F, tp=tp, fp=fp, fn=fn, tn=tn, fpr=fpr)


def macro_prf(df: pd.DataFrame, pred_col: str) -> dict:
    rows = [prf(g.y, g[pred_col]) for _, g in df.groupby("antenna")]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in ("precision", "recall", "f1")}


def prevalence_adjusted_precision(tp: int, fp: int, n_pos: int, n_neg: int, target_prev: float) -> float:
    """Precision the same TPR/FPR would give at `target_prev` (removes the purely
    compositional effect of dropping positives while keeping all negatives)."""
    tpr = tp / n_pos; fpr = fp / n_neg
    return tpr * target_prev / (tpr * target_prev + fpr * (1 - target_prev))


def cluster_ids(df: pd.DataFrame, freq: str = "15min") -> np.ndarray:
    """Cluster = (event bucket) for bursts; each negative is its own cluster.
    Samples of the same physical event are correlated, so resampling must be by cluster."""
    b = bucket(df.start_datetime, freq).astype("int64").values
    cid = np.where(df.y.values == 1, b, -(np.arange(len(df)) + 1))
    _, inv = np.unique(cid, return_inverse=True)
    return inv


def cluster_bootstrap(df: pd.DataFrame, stat_fn, B: int = 2000, seed: int = 42, freq: str = "15min"):
    """Percentile CI of stat_fn(df_resampled) with cluster (event) resampling.
    stat_fn receives an index array into df and returns a float or dict of floats."""
    rng = np.random.default_rng(seed)
    cid = cluster_ids(df, freq)
    n_c = cid.max() + 1
    members = pd.Series(np.arange(len(df))).groupby(cid).apply(np.array).values
    draws = []
    for _ in range(B):
        pick = rng.integers(0, n_c, n_c)
        idx = np.concatenate(members[pick])
        draws.append(stat_fn(idx))
    if isinstance(draws[0], dict):
        return {k: [float(np.nanpercentile([d[k] for d in draws], 2.5)),
                    float(np.nanpercentile([d[k] for d in draws], 97.5))] for k in draws[0]}
    return [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


# ----------------------------------------------------------------------------- io
def save_json(obj, name: str):
    path = RESULTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    print(f"[saved] {path.relative_to(ROOT)}")


def _jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    if isinstance(o, (pd.Timestamp,)): return o.isoformat()
    return str(o)
