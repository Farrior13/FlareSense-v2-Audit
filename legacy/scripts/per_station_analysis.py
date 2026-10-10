#!/usr/bin/env python3
"""Per-station failure mode analysis for FlareSense-v2 audit.

Computes per-station metrics (full vs clean), FP/FN analysis,
and model confidence comparison between leaked and clean bursts.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score


def load_data():
    """Load all splits from HuggingFace."""
    cols = ["manual_label", "prob", "model_label", "start_datetime", "antenna"]
    
    print("Loading dataset from HuggingFace...")
    df_train = load_dataset("i4ds/ecallisto_radio_sunburst", split="train").select_columns(cols).to_pandas()
    df_val = load_dataset("i4ds/ecallisto_radio_sunburst", split="val").select_columns(cols).to_pandas()
    df_test = load_dataset("i4ds/ecallisto_radio_sunburst", split="test").select_columns(cols).to_pandas()
    
    df_tv = pd.concat([df_train, df_val], ignore_index=True)
    df_tv["start_datetime"] = pd.to_datetime(df_tv["start_datetime"])
    df_test["start_datetime"] = pd.to_datetime(df_test["start_datetime"])
    
    return df_tv, df_test


def compute_leaked_mask(df_tv: pd.DataFrame, df_test: pd.DataFrame) -> pd.Series:
    """Identify leaked burst samples in test set using 15-min buckets."""
    tv_burst = df_tv[df_tv["manual_label"] != 0].copy()
    test_all = df_test.copy()
    
    tv_burst["bucket"] = tv_burst["start_datetime"].dt.floor("15min")
    test_all["bucket"] = test_all["start_datetime"].dt.floor("15min")
    
    shared_buckets = set(tv_burst["bucket"].unique()) & set(
        test_all.loc[test_all["manual_label"] != 0, "bucket"].unique()
    )
    
    is_burst = test_all["manual_label"] != 0
    is_leaked = is_burst & test_all["bucket"].isin(shared_buckets)
    
    return is_leaked


def per_station_analysis(df_test: pd.DataFrame, is_leaked: pd.Series):
    """Compute per-station metrics for full and clean test sets."""
    y_true = (df_test["manual_label"] != 0).astype(int).values
    y_pred = df_test["model_label"].values
    stations = sorted(df_test["antenna"].unique())
    
    results = []
    for st in stations:
        mask_st = (df_test["antenna"] == st).values
        yt, yp = y_true[mask_st], y_pred[mask_st]
        n_burst = yt.sum()
        
        if n_burst > 0:
            p_f = precision_score(yt, yp, zero_division=0)
            r_f = recall_score(yt, yp, zero_division=0)
            f_f = f1_score(yt, yp, zero_division=0)
        else:
            p_f = r_f = f_f = float("nan")
        
        clean_mask = mask_st & (~is_leaked.values)
        yt_c, yp_c = y_true[clean_mask], y_pred[clean_mask]
        n_burst_c = yt_c.sum()
        
        if n_burst_c >= 6:
            p_c = precision_score(yt_c, yp_c, zero_division=0)
            r_c = recall_score(yt_c, yp_c, zero_division=0)
            f_c = f1_score(yt_c, yp_c, zero_division=0)
        else:
            p_c = r_c = f_c = float("nan")
        
        results.append({
            "station": st,
            "n_total": int(mask_st.sum()),
            "n_burst_full": int(n_burst),
            "n_burst_clean": int(n_burst_c),
            "f1_full": round(f_f, 4),
            "f1_clean": round(f_c, 4) if not np.isnan(f_c) else None,
            "prec_full": round(p_f, 4),
            "prec_clean": round(p_c, 4) if not np.isnan(p_c) else None,
            "rec_full": round(r_f, 4),
            "rec_clean": round(r_c, 4) if not np.isnan(r_c) else None,
            "delta_f1": round(f_c - f_f, 4) if not np.isnan(f_c) else None,
        })
    
    return sorted(results, key=lambda x: x["f1_full"], reverse=True)


def fp_fn_analysis(df_test: pd.DataFrame):
    """Compute FP and FN distribution by station."""
    y_true = (df_test["manual_label"] != 0).astype(int).values
    y_pred = df_test["model_label"].values
    
    fp_results = []
    fn_results = []
    
    for st in sorted(df_test["antenna"].unique()):
        mask = (df_test["antenna"] == st).values
        yt, yp = y_true[mask], y_pred[mask]
        
        n_neg = (yt == 0).sum()
        n_pos = (yt == 1).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        
        if n_neg > 0:
            fp_results.append({"station": st, "fp": int(fp), "negatives": int(n_neg),
                              "fp_rate": round(fp / n_neg * 100, 2)})
        if n_pos > 0:
            fn_results.append({"station": st, "fn": int(fn), "bursts": int(n_pos),
                              "miss_rate": round(fn / n_pos * 100, 2)})
    
    return (sorted(fp_results, key=lambda x: x["fp_rate"], reverse=True),
            sorted(fn_results, key=lambda x: x["miss_rate"], reverse=True))


def confidence_analysis(df_test: pd.DataFrame, is_leaked: pd.Series) -> dict:
    """Compare model confidence on leaked vs clean burst samples."""
    is_burst = df_test["manual_label"] != 0
    
    leaked_probs = df_test.loc[is_leaked, "prob"].values
    clean_probs = df_test.loc[is_burst & ~is_leaked, "prob"].values
    
    result = {
        "leaked": {
            "n": len(leaked_probs),
            "mean_prob": round(float(np.mean(leaked_probs)), 4),
            "median_prob": round(float(np.median(leaked_probs)), 4),
            "p25": round(float(np.percentile(leaked_probs, 25)), 4),
            "recall": round(float((leaked_probs >= 0.5).mean()), 4),
        },
        "clean": {
            "n": len(clean_probs),
            "mean_prob": round(float(np.mean(clean_probs)), 4),
            "median_prob": round(float(np.median(clean_probs)), 4),
            "p25": round(float(np.percentile(clean_probs, 25)), 4),
            "recall": round(float((clean_probs >= 0.5).mean()), 4),
        },
    }
    result["delta_mean_prob"] = round(result["clean"]["mean_prob"] - result["leaked"]["mean_prob"], 4)
    result["delta_median_prob"] = round(result["clean"]["median_prob"] - result["leaked"]["median_prob"], 4)
    result["delta_recall"] = round(result["clean"]["recall"] - result["leaked"]["recall"], 4)
    
    return result


def print_results(station_results, fp_results, fn_results, conf):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print("PER-STATION METRICS (Full vs Clean)")
    print(f"{'='*80}")
    print(f"{'Station':<30} {'N':>5} {'B_full':>6} {'B_cln':>5} {'F1_full':>7} {'F1_cln':>7} {'dF1':>7}")
    print("-" * 80)
    for r in station_results:
        f1c = f"{r['f1_clean']:.3f}" if r["f1_clean"] is not None else "N/A"
        df1 = f"{r['delta_f1']:+.3f}" if r["delta_f1"] is not None else "N/A"
        print(f"{r['station']:<30} {r['n_total']:>5} {r['n_burst_full']:>6} "
              f"{r['n_burst_clean']:>5} {r['f1_full']:>7.3f} {f1c:>7} {df1:>7}")
    
    print(f"\n{'='*80}")
    print("FALSE POSITIVES BY STATION (top 10)")
    print(f"{'='*80}")
    for r in fp_results[:10]:
        print(f"  {r['station']}: {r['fp']} FP / {r['negatives']} neg ({r['fp_rate']:.2f}%)")
    
    print(f"\n{'='*80}")
    print("FALSE NEGATIVES BY STATION (top 10)")
    print(f"{'='*80}")
    for r in fn_results[:10]:
        print(f"  {r['station']}: {r['fn']} FN / {r['bursts']} bursts ({r['miss_rate']:.2f}%)")
    
    print(f"\n{'='*80}")
    print("MODEL CONFIDENCE: LEAKED vs CLEAN BURSTS")
    print(f"{'='*80}")
    for label in ["leaked", "clean"]:
        c = conf[label]
        print(f"  {label.capitalize()} (n={c['n']}): mean={c['mean_prob']:.4f}, "
              f"median={c['median_prob']:.4f}, P25={c['p25']:.4f}, recall={c['recall']:.4f}")
    print(f"  Delta mean prob:   {conf['delta_mean_prob']:+.4f}")
    print(f"  Delta median prob: {conf['delta_median_prob']:+.4f}")
    print(f"  Delta recall:      {conf['delta_recall']:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Per-station failure mode analysis")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df_tv, df_test = load_data()
    is_leaked = compute_leaked_mask(df_tv, df_test)
    
    station_results = per_station_analysis(df_test, is_leaked)
    fp_results, fn_results = fp_fn_analysis(df_test)
    conf = confidence_analysis(df_test, is_leaked)
    
    print_results(station_results, fp_results, fn_results, conf)
    
    # Save results
    out = Path(args.output_dir)
    with open(out / "per_station_metrics.json", "w") as f:
        json.dump(station_results, f, indent=2)
    with open(out / "confidence_analysis.json", "w") as f:
        json.dump(conf, f, indent=2)
    with open(out / "fp_fn_analysis.json", "w") as f:
        json.dump({"false_positives": fp_results, "false_negatives": fn_results}, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
