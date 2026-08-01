"""
Main entry point script that reproduces ALL results from the FlareSense audit.

This script loads the `i4ds/ecallisto_radio_sunburst` dataset and performs
various evaluations including standard metrics, data leakage analysis via temporal
overlap, clean subset evaluation, bootstrap confidence intervals, and Bayesian PPV.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(message)s")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate basic classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "tpr": float(tpr),
        "fpr": float(fpr),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstraps: int = 10000, seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% confidence intervals using bootstrapping."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    metrics = {"precision": [], "recall": [], "f1": []}
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Avoid zero division warnings in bootstrap if only one class is sampled
        if len(np.unique(y_true_boot)) > 1:
            metrics["precision"].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics["recall"].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics["f1"].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))

    ci_results = {}
    for k, v in metrics.items():
        ci_results[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        
    return ci_results


def bayesian_ppv(tpr: float, fpr: float, pi: float) -> float:
    """Calculate Positive Predictive Value (Precision) given a base rate (pi)."""
    return (tpr * pi) / (tpr * pi + fpr * (1 - pi))


def main(output_dir: str):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    logging.info("1. Loading dataset...")
    # Load dataset
    ds = load_dataset("i4ds/ecallisto_radio_sunburst")
    
    columns = ["manual_label", "prob", "model_label", "start_datetime", "antenna"]
    
    train_df = ds["train"].to_pandas()[columns]
    val_df = ds["val"].to_pandas()[columns]
    test_df = ds["test"].to_pandas()[columns]
    
    # 3a. Dataset sizes
    logging.info("3a. Dataset sizes:")
    logging.info(f"  Train: {len(train_df)}")
    logging.info(f"  Val:   {len(val_df)}")
    logging.info(f"  Test:  {len(test_df)}")
    
    results["dataset_sizes"] = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df)
    }

    # 3b-d. Full test metrics
    y_test_true = test_df["manual_label"].values
    y_test_pred = test_df["model_label"].values
    
    full_metrics = calculate_metrics(y_test_true, y_test_pred)
    logging.info("\n3b-d. Full test metrics:")
    logging.info(f"  Precision: {full_metrics['precision']*100:.2f}%")
    logging.info(f"  Recall:    {full_metrics['recall']*100:.2f}%")
    logging.info(f"  F1:        {full_metrics['f1']*100:.2f}%")
    logging.info(f"  Confusion Matrix: TP={full_metrics['tp']}, FP={full_metrics['fp']}, "
                 f"FN={full_metrics['fn']}, TN={full_metrics['tn']}")
    logging.info(f"  TPR={full_metrics['tpr']:.4f}, FPR={full_metrics['fpr']:.4f}")
    
    results["full_test_metrics"] = full_metrics

    # 4. Event overlap analysis
    logging.info("\n4. Event overlap analysis:")
    # Convert datetimes
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    
    train_val_df["start_datetime"] = pd.to_datetime(train_val_df["start_datetime"])
    test_df["start_datetime"] = pd.to_datetime(test_df["start_datetime"])
    
    # Filter bursts (manual_label == 1)
    tv_bursts = train_val_df[train_val_df["manual_label"] != 0]
    test_bursts = test_df[test_df["manual_label"] != 0]
    
    # 15-min buckets
    tv_15m = tv_bursts["start_datetime"].dt.floor("15min").unique()
    test_bursts_15m_floor = test_bursts["start_datetime"].dt.floor("15min")
    leaked_15m_mask = test_bursts_15m_floor.isin(tv_15m)
    
    leaked_15m = leaked_15m_mask.sum()
    clean_15m = (~leaked_15m_mask).sum()
    total_test_bursts = len(test_bursts)
    pct_15m = leaked_15m / total_test_bursts * 100 if total_test_bursts else 0
    
    logging.info(f"  15-min leaked: {leaked_15m} leaked, {clean_15m} clean, {pct_15m:.1f}%")
    
    # 1-hour buckets
    tv_1h = tv_bursts["start_datetime"].dt.floor("1h").unique()
    test_bursts_1h_floor = test_bursts["start_datetime"].dt.floor("1h")
    leaked_1h_mask = test_bursts_1h_floor.isin(tv_1h)
    
    leaked_1h = leaked_1h_mask.sum()
    clean_1h = (~leaked_1h_mask).sum()
    pct_1h = leaked_1h / total_test_bursts * 100 if total_test_bursts else 0
    
    logging.info(f"  1-hour leaked: {leaked_1h} leaked, {clean_1h} clean, {pct_1h:.1f}%")
    
    results["overlap_analysis"] = {
        "15min": {"leaked": int(leaked_15m), "clean": int(clean_15m), "pct": float(pct_15m)},
        "1h": {"leaked": int(leaked_1h), "clean": int(clean_1h), "pct": float(pct_1h)}
    }

    # 5. Clean evaluation
    logging.info("\n5. Clean evaluation (15m):")
    
    # Identify clean subset: non-bursts OR clean bursts
    test_df_w_leak_flag = test_df.copy()
    test_df_w_leak_flag["15m_floor"] = test_df_w_leak_flag["start_datetime"].dt.floor("15min")
    
    is_burst = test_df_w_leak_flag["manual_label"] != 0
    is_leaked = test_df_w_leak_flag["15m_floor"].isin(tv_15m) & is_burst
    
    clean_test_df = test_df_w_leak_flag[~is_leaked]
    
    y_clean_true = clean_test_df["manual_label"].values
    y_clean_pred = clean_test_df["model_label"].values
    
    clean_metrics = calculate_metrics(y_clean_true, y_clean_pred)
    logging.info(f"  Precision: {clean_metrics['precision']*100:.2f}%")
    logging.info(f"  Recall:    {clean_metrics['recall']*100:.2f}%")
    logging.info(f"  F1:        {clean_metrics['f1']*100:.2f}%")
    
    results["clean_15m_metrics"] = clean_metrics

    # 6. Bootstrap CI
    logging.info("\n6. Bootstrap CI (B=10000, seed=42):")
    full_ci = compute_bootstrap_ci(y_test_true, y_test_pred)
    clean_ci = compute_bootstrap_ci(y_clean_true, y_clean_pred)
    
    logging.info("  Full test CI:")
    logging.info(f"    Precision: [{full_ci['precision'][0]:.4f}, {full_ci['precision'][1]:.4f}]")
    logging.info(f"    Recall:    [{full_ci['recall'][0]:.4f}, {full_ci['recall'][1]:.4f}]")
    logging.info(f"    F1:        [{full_ci['f1'][0]:.4f}, {full_ci['f1'][1]:.4f}]")
    
    logging.info("  Clean 15m CI:")
    logging.info(f"    Precision: [{clean_ci['precision'][0]:.4f}, {clean_ci['precision'][1]:.4f}]")
    logging.info(f"    Recall:    [{clean_ci['recall'][0]:.4f}, {clean_ci['recall'][1]:.4f}]")
    logging.info(f"    F1:        [{clean_ci['f1'][0]:.4f}, {clean_ci['f1'][1]:.4f}]")
    
    results["bootstrap_ci"] = {
        "full": full_ci,
        "clean_15m": clean_ci
    }

    # 7. Confidence analysis
    logging.info("\n7. Confidence analysis:")
    
    leaked_bursts_df = test_bursts[leaked_15m_mask]
    clean_bursts_df = test_bursts[~leaked_15m_mask]
    
    def get_stats(df):
        if len(df) == 0:
            return {"median_prob": 0, "mean_prob": 0, "p25": 0, "recall": 0}
        return {
            "median_prob": float(df["prob"].median()),
            "mean_prob": float(df["prob"].mean()),
            "p25": float(df["prob"].quantile(0.25)),
            "recall": float((df["model_label"] == 1).mean())
        }
        
    leaked_stats = get_stats(leaked_bursts_df)
    clean_stats = get_stats(clean_bursts_df)
    
    logging.info("  Leaked bursts:")
    logging.info(f"    Median prob: {leaked_stats['median_prob']:.4f}")
    logging.info(f"    Mean prob:   {leaked_stats['mean_prob']:.4f}")
    logging.info(f"    P25:         {leaked_stats['p25']:.4f}")
    logging.info(f"    Recall:      {leaked_stats['recall']:.4f}")
    
    logging.info("  Clean bursts:")
    logging.info(f"    Median prob: {clean_stats['median_prob']:.4f}")
    logging.info(f"    Mean prob:   {clean_stats['mean_prob']:.4f}")
    logging.info(f"    P25:         {clean_stats['p25']:.4f}")
    logging.info(f"    Recall:      {clean_stats['recall']:.4f}")
    
    logging.info("  Deltas (Leaked - Clean):")
    logging.info(f"    Median prob: {leaked_stats['median_prob'] - clean_stats['median_prob']:.4f}")
    logging.info(f"    Mean prob:   {leaked_stats['mean_prob'] - clean_stats['mean_prob']:.4f}")
    logging.info(f"    P25:         {leaked_stats['p25'] - clean_stats['p25']:.4f}")
    logging.info(f"    Recall:      {leaked_stats['recall'] - clean_stats['recall']:.4f}")
    
    results["confidence_analysis"] = {
        "leaked": leaked_stats,
        "clean": clean_stats,
        "deltas": {k: leaked_stats[k] - clean_stats[k] for k in leaked_stats}
    }

    # 8. Bayesian PPV calculation
    logging.info("\n8. Bayesian PPV:")
    
    pi_values = [0.10, 0.01, 0.001, 0.0001]
    ppv_results = {}
    
    # Use full test TPR/FPR (as in the audit report)
    tpr = full_metrics["tpr"]
    fpr = full_metrics["fpr"]
    
    for pi in pi_values:
        ppv = bayesian_ppv(tpr, fpr, pi)
        ppv_results[f"{pi}"] = ppv
        logging.info(f"  pi={pi:<6}: PPV = {ppv*100:.2f}%")
        
    results["bayesian_ppv"] = ppv_results

    # 9. Save results
    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"\nResults saved to {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce all results from the audit.")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to save output files")
    args = parser.parse_args()
    
    main(args.output_dir)
