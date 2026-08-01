#!/usr/bin/env python3
"""Generate all 6 publication-ready figures for the FlareSense-v2 audit.

All figures are computed from the actual HuggingFace dataset predictions,
not synthetic data.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score


def load_and_prepare():
    """Load dataset and compute overlap masks."""
    print("Loading dataset from HuggingFace...")
    cols = ["manual_label", "prob", "model_label", "start_datetime", "antenna"]

    df_train = load_dataset("i4ds/ecallisto_radio_sunburst", split="train").select_columns(cols).to_pandas()
    df_val = load_dataset("i4ds/ecallisto_radio_sunburst", split="val").select_columns(cols).to_pandas()
    df_test = load_dataset("i4ds/ecallisto_radio_sunburst", split="test").select_columns(cols).to_pandas()

    df_tv = pd.concat([df_train, df_val], ignore_index=True)
    df_tv["start_datetime"] = pd.to_datetime(df_tv["start_datetime"])
    df_test["start_datetime"] = pd.to_datetime(df_test["start_datetime"])

    # Compute 15-min overlap
    tv_burst = df_tv[df_tv["manual_label"] != 0].copy()
    tv_burst["bucket"] = tv_burst["start_datetime"].dt.floor("15min")
    df_test["bucket"] = df_test["start_datetime"].dt.floor("15min")
    test_burst_buckets = df_test.loc[df_test["manual_label"] != 0, "bucket"].unique()
    shared_15m = set(tv_burst["bucket"].unique()) & set(test_burst_buckets)

    # Compute 1-hour overlap
    tv_burst["bucket_1h"] = tv_burst["start_datetime"].dt.floor("1h")
    df_test["bucket_1h"] = df_test["start_datetime"].dt.floor("1h")
    test_burst_buckets_1h = df_test.loc[df_test["manual_label"] != 0, "bucket_1h"].unique()
    shared_1h = set(tv_burst["bucket_1h"].unique()) & set(test_burst_buckets_1h)

    is_burst = df_test["manual_label"] != 0
    is_leaked_15m = is_burst & df_test["bucket"].isin(shared_15m)
    is_leaked_1h = is_burst & df_test["bucket_1h"].isin(shared_1h)

    return df_test, is_burst, is_leaked_15m, is_leaked_1h


def bootstrap_ci(y_true, y_pred, metric_fn, B=10000, seed=42):
    """Compute bootstrap 95% CI for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def fig1_negative_prob_dist(df_test, is_burst, out_dir):
    """Histogram of predicted probabilities for non-burst samples."""
    neg_probs = df_test.loc[~is_burst, "prob"].values
    median_val = np.median(neg_probs)
    p95_val = np.percentile(neg_probs, 95)
    pct_below_005 = (neg_probs < 0.05).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(neg_probs, bins=100, color="#2C8C99", alpha=0.8, log=True)
    ax.axvline(median_val, color="#E74C3C", linestyle="--", linewidth=2,
               label=f"Median = {median_val:.3f}")
    ax.axvline(p95_val, color="#F39C12", linestyle="--", linewidth=2,
               label=f"P95 = {p95_val:.3f}")
    ax.axvline(0.05, color="black", linestyle=":", linewidth=1.5,
               label=f"prob = 0.05 ({pct_below_005:.1f}% below)")
    ax.set_xlabel("Predicted Burst Probability", fontsize=13)
    ax.set_ylabel("Count (log scale)", fontsize=13)
    ax.set_title("Predicted probabilities for non-burst test samples", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_negative_prob_dist.png"), dpi=150)
    plt.close()
    print("  Fig 1 saved.")


def fig2_burst_vs_nonburst(df_test, is_burst, out_dir):
    """Overlapping histograms of burst vs non-burst probability distributions."""
    burst_probs = df_test.loc[is_burst, "prob"].values
    neg_probs = df_test.loc[~is_burst, "prob"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(neg_probs, bins=50, alpha=0.6, color="#3498DB", density=True,
            label=f"Non-burst (n={len(neg_probs):,})")
    ax.hist(burst_probs, bins=50, alpha=0.6, color="#E67E22", density=True,
            label=f"Burst (n={len(burst_probs):,})")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted Burst Probability", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Burst vs non-burst probability distributions", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_burst_vs_nonburst.png"), dpi=150)
    plt.close()
    print("  Fig 2 saved.")


def fig3_base_rate_ppv(out_dir):
    """PPV vs prevalence curve using observed TPR and FPR."""
    tpr, fpr = 0.7990, 0.0129
    pi = np.logspace(-4, np.log10(0.5), 500)
    ppv = (tpr * pi) / (tpr * pi + fpr * (1 - pi))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pi, ppv * 100, color="#2C3E50", linewidth=2.5)
    ax.axhline(90.6, color="#BDC3C7", linestyle="--", linewidth=1,
               label="Reported precision (90.6%)")

    key_points = [(0.10, "87.4%"), (0.01, "38.6%"), (0.001, "5.9%"), (0.0001, "0.6%")]
    for p, label in key_points:
        val = (tpr * p) / (tpr * p + fpr * (1 - p)) * 100
        ax.scatter([p], [val], color="#E74C3C", s=80, zorder=5)
        ax.annotate(f"π={p}\nPPV={label}", (p, val),
                    textcoords="offset points", xytext=(12, -5),
                    fontsize=10, ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("Prevalence (π)", fontsize=13)
    ax.set_ylabel("Positive Predictive Value (%)", fontsize=13)
    ax.set_title("PPV vs base rate (Bayes' theorem)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_base_rate_ppv.png"), dpi=150)
    plt.close()
    print("  Fig 3 saved.")


def fig4_metrics_comparison(df_test, is_burst, is_leaked_15m, is_leaked_1h, out_dir):
    """Grouped bar chart: Full vs Clean 15m vs Clean 1h with bootstrap CIs."""
    y_true = (df_test["manual_label"] != 0).astype(int).values
    y_pred = df_test["model_label"].values

    # Full test
    p_f = precision_score(y_true, y_pred)
    r_f = recall_score(y_true, y_pred)
    f_f = f1_score(y_true, y_pred)
    p_f_ci = bootstrap_ci(y_true, y_pred, precision_score)
    r_f_ci = bootstrap_ci(y_true, y_pred, recall_score)
    f_f_ci = bootstrap_ci(y_true, y_pred, f1_score)

    # Clean 15m
    mask_c15 = ~is_leaked_15m.values
    yt_c15, yp_c15 = y_true[mask_c15], y_pred[mask_c15]
    p_c15 = precision_score(yt_c15, yp_c15)
    r_c15 = recall_score(yt_c15, yp_c15)
    f_c15 = f1_score(yt_c15, yp_c15)
    p_c15_ci = bootstrap_ci(yt_c15, yp_c15, precision_score)
    r_c15_ci = bootstrap_ci(yt_c15, yp_c15, recall_score)
    f_c15_ci = bootstrap_ci(yt_c15, yp_c15, f1_score)

    # Clean 1h
    mask_c1h = ~is_leaked_1h.values
    yt_c1h, yp_c1h = y_true[mask_c1h], y_pred[mask_c1h]
    p_c1h = precision_score(yt_c1h, yp_c1h)
    r_c1h = recall_score(yt_c1h, yp_c1h)
    f_c1h = f1_score(yt_c1h, yp_c1h)
    p_c1h_ci = bootstrap_ci(yt_c1h, yp_c1h, precision_score)
    r_c1h_ci = bootstrap_ci(yt_c1h, yp_c1h, recall_score)
    f_c1h_ci = bootstrap_ci(yt_c1h, yp_c1h, f1_score)

    metrics = ["Precision", "Recall", "F1"]
    full_vals = [p_f, r_f, f_f]
    c15_vals = [p_c15, r_c15, f_c15]
    c1h_vals = [p_c1h, r_c1h, f_c1h]

    full_errs = [[p_f - p_f_ci[0], r_f - r_f_ci[0], f_f - f_f_ci[0]],
                 [p_f_ci[1] - p_f, r_f_ci[1] - r_f, f_f_ci[1] - f_f]]
    c15_errs = [[p_c15 - p_c15_ci[0], r_c15 - r_c15_ci[0], f_c15 - f_c15_ci[0]],
                [p_c15_ci[1] - p_c15, r_c15_ci[1] - r_c15, f_c15_ci[1] - f_c15]]
    c1h_errs = [[p_c1h - p_c1h_ci[0], r_c1h - r_c1h_ci[0], f_c1h - f_c1h_ci[0]],
                [p_c1h_ci[1] - p_c1h, r_c1h_ci[1] - r_c1h, f_c1h_ci[1] - f_c1h]]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, full_vals, width, yerr=full_errs, capsize=4,
           color="#2C3E50", alpha=0.85, label="Full test")
    ax.bar(x, c15_vals, width, yerr=c15_errs, capsize=4,
           color="#E67E22", alpha=0.85, label="Clean 15m")
    ax.bar(x + width, c1h_vals, width, yerr=c1h_errs, capsize=4,
           color="#E74C3C", alpha=0.85, label="Clean 1h")

    ax.set_ylabel("Score", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Metrics: full test vs event-disjoint subsets", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for i, (fv, c15v, c1hv) in enumerate(zip(full_vals, c15_vals, c1h_vals)):
        ax.text(i - width, fv + 0.01, f"{fv:.1%}", ha="center", fontsize=9)
        ax.text(i, c15v + 0.01, f"{c15v:.1%}", ha="center", fontsize=9)
        ax.text(i + width, c1hv + 0.01, f"{c1hv:.1%}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_metrics_comparison.png"), dpi=150)
    plt.close()
    print("  Fig 4 saved.")


def fig5_leaked_vs_clean_prob(df_test, is_burst, is_leaked_15m, out_dir):
    """Overlapping histograms: leaked vs clean burst predicted probabilities."""
    leaked_probs = df_test.loc[is_leaked_15m, "prob"].values
    clean_probs = df_test.loc[is_burst & ~is_leaked_15m, "prob"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(leaked_probs, bins=50, alpha=0.6, color="#E74C3C", density=True,
            label=f"Leaked bursts (n={len(leaked_probs):,})")
    ax.hist(clean_probs, bins=50, alpha=0.6, color="#2ECC71", density=True,
            label=f"Clean bursts (n={len(clean_probs):,})")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="Threshold")
    ax.axvline(np.median(leaked_probs), color="#C0392B", linestyle=":",
               linewidth=2, label=f"Leaked median = {np.median(leaked_probs):.3f}")
    ax.axvline(np.median(clean_probs), color="#27AE60", linestyle=":",
               linewidth=2, label=f"Clean median = {np.median(clean_probs):.3f}")
    ax.set_xlabel("Predicted Burst Probability", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Model confidence: leaked vs clean burst samples", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_leaked_vs_clean_prob.png"), dpi=150)
    plt.close()
    print("  Fig 5 saved.")


def fig6_per_station_delta_f1(df_test, is_burst, is_leaked_15m, out_dir):
    """Horizontal bar chart of ΔF1 per station."""
    y_true = (df_test["manual_label"] != 0).astype(int).values
    y_pred = df_test["model_label"].values
    stations = sorted(df_test["antenna"].unique())

    results = []
    for st in stations:
        mask_st = (df_test["antenna"] == st).values
        yt, yp = y_true[mask_st], y_pred[mask_st]
        n_burst = yt.sum()
        if n_burst == 0:
            continue
        f1_full = f1_score(yt, yp, zero_division=0)

        clean_mask = mask_st & (~is_leaked_15m.values)
        yt_c, yp_c = y_true[clean_mask], y_pred[clean_mask]
        if yt_c.sum() < 6:
            continue
        f1_clean = f1_score(yt_c, yp_c, zero_division=0)
        results.append({"station": st, "f1_full": f1_full,
                        "f1_clean": f1_clean, "delta": f1_clean - f1_full})

    results.sort(key=lambda x: x["delta"])

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#E74C3C" if r["delta"] < -0.05
              else "#F39C12" if r["delta"] < 0
              else "#2ECC71" for r in results]
    y_pos = range(len(results))
    deltas = [r["delta"] for r in results]
    labels = [r["station"] for r in results]

    ax.barh(y_pos, deltas, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("ΔF1 (clean − full)", fontsize=13)
    ax.set_title("Per-station F1 change after removing leaked events", fontsize=14)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    for i, r in enumerate(results):
        offset = -0.008 if r["delta"] < 0 else 0.008
        ha = "right" if r["delta"] < 0 else "left"
        ax.text(r["delta"] + offset, i, f"{r['delta']:+.3f}",
                va="center", ha=ha, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_per_station_delta_f1.png"), dpi=150)
    plt.close()
    print("  Fig 6 saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all 6 audit figures from HuggingFace data")
    parser.add_argument("--output-dir", default="figures",
                        help="Output directory (default: figures/)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_test, is_burst, is_leaked_15m, is_leaked_1h = load_and_prepare()

    print("Generating figures...")
    fig1_negative_prob_dist(df_test, is_burst, args.output_dir)
    fig2_burst_vs_nonburst(df_test, is_burst, args.output_dir)
    fig3_base_rate_ppv(args.output_dir)
    fig4_metrics_comparison(df_test, is_burst, is_leaked_15m, is_leaked_1h, args.output_dir)
    fig5_leaked_vs_clean_prob(df_test, is_burst, is_leaked_15m, args.output_dir)
    fig6_per_station_delta_f1(df_test, is_burst, is_leaked_15m, args.output_dir)
    print(f"\nAll 6 figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
