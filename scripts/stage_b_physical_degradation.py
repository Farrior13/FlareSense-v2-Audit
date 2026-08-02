"""
Stage B: Physical degradation analysis after leakage removal.

Question: "What exactly does the model stop being able to do
after removing leaked events?"

Physical proxies (since we have only binary labels):
1. Event size: number of stations observing the same event
2. Event duration: time span across stations
3. Model confidence: probability assigned by the model
4. Station geography: per-station degradation
5. Temporal pattern: time-of-day / solar context
"""

import pandas as pd
import numpy as np
from datasets import load_dataset

print("Loading dataset...")
cols = ["manual_label", "prob", "model_label", "start_datetime", "antenna"]

ds_train = load_dataset("i4ds/ecallisto_radio_sunburst", split="train")
ds_val = load_dataset("i4ds/ecallisto_radio_sunburst", split="val")
ds_test = load_dataset("i4ds/ecallisto_radio_sunburst", split="test")

df_tv = pd.concat([
    ds_train.select_columns(cols).to_pandas(),
    ds_val.select_columns(cols).to_pandas()
])
df_test = ds_test.select_columns(cols).to_pandas()

df_tv["start_datetime"] = pd.to_datetime(df_tv["start_datetime"])
df_test["start_datetime"] = pd.to_datetime(df_test["start_datetime"])

# Check manual_label distribution
print("\n=== MANUAL LABEL DISTRIBUTION ===")
print("Train+Val:")
print(df_tv["manual_label"].value_counts().sort_index())
print("\nTest:")
print(df_test["manual_label"].value_counts().sort_index())

# === 1. Compute event overlap (15-min buckets) ===
print("\n=== COMPUTING EVENT OVERLAP ===")
tv_burst = df_tv[df_tv["manual_label"] != 0].copy()
test_burst = df_test[df_test["manual_label"] != 0].copy()
tv_burst["bucket"] = tv_burst["start_datetime"].dt.floor("15min")
test_burst["bucket"] = test_burst["start_datetime"].dt.floor("15min")

shared_buckets = set(tv_burst["bucket"]) & set(test_burst["bucket"])
test_burst["is_leaked"] = test_burst["bucket"].isin(shared_buckets)

print(f"Total test bursts: {len(test_burst)}")
print(f"Leaked bursts: {test_burst['is_leaked'].sum()}")
print(f"Clean bursts: {(~test_burst['is_leaked']).sum()}")

# === 2. Event characteristics ===
print("\n=== COMPUTING EVENT CHARACTERISTICS ===")

# For each bucket, count stations in train+val and test
all_burst = pd.concat([tv_burst, test_burst[["start_datetime", "antenna", "bucket", "manual_label"]]])
event_station_count = all_burst.groupby("bucket")["antenna"].nunique().rename("total_stations")
event_duration = all_burst.groupby("bucket").agg(
    start=("start_datetime", "min"),
    end=("start_datetime", "max")
)
event_duration["duration_min"] = (event_duration["end"] - event_duration["start"]).dt.total_seconds() / 60

# How many train stations per event
tv_stations_per_bucket = tv_burst.groupby("bucket")["antenna"].nunique().rename("train_stations")

# Merge into test_burst
test_burst = test_burst.merge(event_station_count, left_on="bucket", right_index=True, how="left")
test_burst = test_burst.merge(tv_stations_per_bucket, left_on="bucket", right_index=True, how="left")
test_burst["train_stations"] = test_burst["train_stations"].fillna(0).astype(int)
test_burst = test_burst.merge(event_duration[["duration_min"]], left_on="bucket", right_index=True, how="left")

# Model predictions
test_burst["predicted_burst"] = (test_burst["prob"] >= 0.5).astype(int)
test_burst["correct"] = (test_burst["predicted_burst"] == 1)  # all are bursts

# Time of day
test_burst["hour_utc"] = test_burst["start_datetime"].dt.hour

# === 3. ANALYSIS: Recall by event size ===
print("\n" + "="*60)
print("ANALYSIS 1: RECALL BY EVENT SIZE (number of observing stations)")
print("="*60)

# Bin by total stations
bins_stations = [0, 1, 2, 3, 5, 10, 30]
labels_stations = ["1 station", "2 stations", "3 stations", "4-5 stations", "6-10 stations", "11+ stations"]
test_burst["station_bin"] = pd.cut(test_burst["total_stations"], bins=bins_stations, labels=labels_stations)

for is_leaked in [True, False]:
    subset = test_burst[test_burst["is_leaked"] == is_leaked]
    label = "LEAKED" if is_leaked else "CLEAN"
    print(f"\n{label}:")
    for bin_label in labels_stations:
        bin_data = subset[subset["station_bin"] == bin_label]
        if len(bin_data) > 0:
            recall = bin_data["correct"].mean()
            mean_prob = bin_data["prob"].mean()
            print(f"  {bin_label:15s}: n={len(bin_data):5d}, Recall={recall:.1%}, Mean prob={mean_prob:.3f}")

# Combined table
print("\n--- DELTA TABLE ---")
print(f"{'Station bin':15s} | {'n_leaked':>8s} | {'n_clean':>7s} | {'R_leaked':>8s} | {'R_clean':>7s} | {'ΔRecall':>7s} | {'ΔProb':>6s}")
print("-" * 75)
for bin_label in labels_stations:
    leaked = test_burst[(test_burst["station_bin"] == bin_label) & test_burst["is_leaked"]]
    clean = test_burst[(test_burst["station_bin"] == bin_label) & ~test_burst["is_leaked"]]
    if len(leaked) > 0 or len(clean) > 0:
        r_l = leaked["correct"].mean() if len(leaked) > 0 else float('nan')
        r_c = clean["correct"].mean() if len(clean) > 0 else float('nan')
        p_l = leaked["prob"].mean() if len(leaked) > 0 else float('nan')
        p_c = clean["prob"].mean() if len(clean) > 0 else float('nan')
        delta_r = r_l - r_c if not (np.isnan(r_l) or np.isnan(r_c)) else float('nan')
        delta_p = p_l - p_c if not (np.isnan(p_l) or np.isnan(p_c)) else float('nan')
        print(f"{bin_label:15s} | {len(leaked):8d} | {len(clean):7d} | {r_l:8.1%} | {r_c:7.1%} | {delta_r:+7.1%} | {delta_p:+6.3f}")

# === 4. ANALYSIS: Recall by event duration ===
print("\n" + "="*60)
print("ANALYSIS 2: RECALL BY EVENT DURATION")
print("="*60)

bins_dur = [-1, 0, 15, 30, 60, 999999]
labels_dur = ["Instantaneous (0m)", "Short (1-15m)", "Medium (16-30m)", "Long (31-60m)", "Very long (60m+)"]
test_burst["duration_bin"] = pd.cut(test_burst["duration_min"], bins=bins_dur, labels=labels_dur)

print(f"\n{'Duration':20s} | {'n_leaked':>8s} | {'n_clean':>7s} | {'R_leaked':>8s} | {'R_clean':>7s} | {'ΔRecall':>7s}")
print("-" * 70)
for bin_label in labels_dur:
    leaked = test_burst[(test_burst["duration_bin"] == bin_label) & test_burst["is_leaked"]]
    clean = test_burst[(test_burst["duration_bin"] == bin_label) & ~test_burst["is_leaked"]]
    if len(leaked) > 0 or len(clean) > 0:
        r_l = leaked["correct"].mean() if len(leaked) > 0 else float('nan')
        r_c = clean["correct"].mean() if len(clean) > 0 else float('nan')
        delta = r_l - r_c if not (np.isnan(r_l) or np.isnan(r_c)) else float('nan')
        print(f"{bin_label:20s} | {len(leaked):8d} | {len(clean):7d} | {r_l:8.1%} | {r_c:7.1%} | {delta:+7.1%}")

# === 5. ANALYSIS: Recall by model confidence ===
print("\n" + "="*60)
print("ANALYSIS 3: CONFIDENCE DISTRIBUTION (leaked vs clean)")
print("="*60)

for is_leaked in [True, False]:
    subset = test_burst[test_burst["is_leaked"] == is_leaked]
    label = "LEAKED" if is_leaked else "CLEAN"
    print(f"\n{label} (n={len(subset)}):")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        print(f"  P{int(q*100):02d}: {subset['prob'].quantile(q):.3f}")
    print(f"  Mean: {subset['prob'].mean():.3f}")
    print(f"  Recall (>=0.5): {(subset['prob'] >= 0.5).mean():.1%}")

# Confidence bins: what fraction of low-confidence predictions are clean?
print("\n--- LOW-CONFIDENCE ANALYSIS ---")
bins_conf = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
labels_conf = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
test_burst["conf_bin"] = pd.cut(test_burst["prob"], bins=bins_conf, labels=labels_conf)

print(f"\n{'Confidence':12s} | {'n_leaked':>8s} | {'n_clean':>7s} | {'%clean':>7s} | {'R_leaked':>8s} | {'R_clean':>7s}")
print("-" * 65)
for bin_label in labels_conf:
    leaked = test_burst[(test_burst["conf_bin"] == bin_label) & test_burst["is_leaked"]]
    clean = test_burst[(test_burst["conf_bin"] == bin_label) & ~test_burst["is_leaked"]]
    total = len(leaked) + len(clean)
    pct_clean = len(clean) / total * 100 if total > 0 else 0
    r_l = leaked["correct"].mean() if len(leaked) > 0 else float('nan')
    r_c = clean["correct"].mean() if len(clean) > 0 else float('nan')
    print(f"{bin_label:12s} | {len(leaked):8d} | {len(clean):7d} | {pct_clean:6.1f}% | {r_l:8.1%} | {r_c:7.1%}")

# === 6. ANALYSIS: Recall by number of train stations (exposure) ===
print("\n" + "="*60)
print("ANALYSIS 4: RECALL BY TRAINING EXPOSURE (train stations per event)")
print("="*60)

bins_train = [-1, 0, 2, 5, 10, 30]
labels_train = ["0 (clean)", "1-2 train stations", "3-5 train stations", "6-10 train stations", "11+ train stations"]
test_burst["train_exposure"] = pd.cut(test_burst["train_stations"], bins=bins_train, labels=labels_train)

print(f"\n{'Training exposure':22s} | {'n':>6s} | {'Recall':>7s} | {'Mean prob':>9s} | {'Median prob':>11s}")
print("-" * 70)
for bin_label in labels_train:
    subset = test_burst[test_burst["train_exposure"] == bin_label]
    if len(subset) > 0:
        recall = subset["correct"].mean()
        mean_p = subset["prob"].mean()
        med_p = subset["prob"].median()
        print(f"{bin_label:22s} | {len(subset):6d} | {recall:7.1%} | {mean_p:9.3f} | {med_p:11.3f}")

# === 7. ANALYSIS: Time of day ===
print("\n" + "="*60)
print("ANALYSIS 5: RECALL BY TIME OF DAY (UTC)")
print("="*60)

bins_hour = [0, 6, 12, 18, 24]
labels_hour = ["00-06 (night)", "06-12 (morning)", "12-18 (afternoon)", "18-24 (evening)"]
test_burst["time_bin"] = pd.cut(test_burst["hour_utc"], bins=bins_hour, labels=labels_hour, right=False)

print(f"\n{'Time (UTC)':20s} | {'n_leaked':>8s} | {'n_clean':>7s} | {'R_leaked':>8s} | {'R_clean':>7s} | {'ΔRecall':>7s}")
print("-" * 70)
for bin_label in labels_hour:
    leaked = test_burst[(test_burst["time_bin"] == bin_label) & test_burst["is_leaked"]]
    clean = test_burst[(test_burst["time_bin"] == bin_label) & ~test_burst["is_leaked"]]
    if len(leaked) > 0 or len(clean) > 0:
        r_l = leaked["correct"].mean() if len(leaked) > 0 else float('nan')
        r_c = clean["correct"].mean() if len(clean) > 0 else float('nan')
        delta = r_l - r_c if not (np.isnan(r_l) or np.isnan(r_c)) else float('nan')
        print(f"{bin_label:20s} | {len(leaked):8d} | {len(clean):7d} | {r_l:8.1%} | {r_c:7.1%} | {delta:+7.1%}")

# === 8. ANALYSIS: Per-station deep dive ===
print("\n" + "="*60)
print("ANALYSIS 6: PER-STATION DEGRADATION (top 15 by burst count)")
print("="*60)

station_stats = []
for station in test_burst["antenna"].unique():
    s_data = test_burst[test_burst["antenna"] == station]
    s_leaked = s_data[s_data["is_leaked"]]
    s_clean = s_data[~s_data["is_leaked"]]
    
    if len(s_leaked) >= 5 and len(s_clean) >= 5:
        r_l = s_leaked["correct"].mean()
        r_c = s_clean["correct"].mean()
        p_l = s_leaked["prob"].mean()
        p_c = s_clean["prob"].mean()
        station_stats.append({
            "station": station,
            "n_leaked": len(s_leaked),
            "n_clean": len(s_clean),
            "recall_leaked": r_l,
            "recall_clean": r_c,
            "delta_recall": r_l - r_c,
            "prob_leaked": p_l,
            "prob_clean": p_c,
            "delta_prob": p_l - p_c
        })

station_df = pd.DataFrame(station_stats).sort_values("delta_recall", ascending=True)

print(f"\n{'Station':25s} | {'n_L':>4s} | {'n_C':>4s} | {'R_L':>6s} | {'R_C':>6s} | {'ΔR':>7s} | {'P_L':>5s} | {'P_C':>5s} | {'ΔP':>6s}")
print("-" * 90)
for _, row in station_df.head(15).iterrows():
    print(f"{row['station']:25s} | {row['n_leaked']:4.0f} | {row['n_clean']:4.0f} | {row['recall_leaked']:6.1%} | {row['recall_clean']:6.1%} | {row['delta_recall']:+7.1%} | {row['prob_leaked']:5.3f} | {row['prob_clean']:5.3f} | {row['delta_prob']:+6.3f}")

# === 9. MANUAL_LABEL type analysis ===
print("\n" + "="*60)
print("ANALYSIS 7: MANUAL LABEL VALUES (burst subtypes?)")
print("="*60)

unique_labels = sorted(test_burst["manual_label"].unique())
print(f"Unique manual_label values in test bursts: {unique_labels}")

if len(unique_labels) > 1:
    print(f"\n{'Label':>6s} | {'n_leaked':>8s} | {'n_clean':>7s} | {'R_leaked':>8s} | {'R_clean':>7s} | {'ΔRecall':>7s}")
    print("-" * 55)
    for label in unique_labels:
        leaked = test_burst[(test_burst["manual_label"] == label) & test_burst["is_leaked"]]
        clean = test_burst[(test_burst["manual_label"] == label) & ~test_burst["is_leaked"]]
        if len(leaked) > 0 or len(clean) > 0:
            r_l = leaked["correct"].mean() if len(leaked) > 0 else float('nan')
            r_c = clean["correct"].mean() if len(clean) > 0 else float('nan')
            delta = r_l - r_c if not (np.isnan(r_l) or np.isnan(r_c)) else float('nan')
            print(f"{label:6d} | {len(leaked):8d} | {len(clean):7d} | {r_l:8.1%} | {r_c:7.1%} | {delta:+7.1%}")

# === SUMMARY ===
print("\n" + "="*60)
print("SUMMARY: KEY FINDINGS")
print("="*60)

# Overall
print(f"\nOverall Recall: Leaked={test_burst[test_burst['is_leaked']]['correct'].mean():.1%}, "
      f"Clean={test_burst[~test_burst['is_leaked']]['correct'].mean():.1%}")
print(f"Overall Prob:   Leaked={test_burst[test_burst['is_leaked']]['prob'].mean():.3f}, "
      f"Clean={test_burst[~test_burst['is_leaked']]['prob'].mean():.3f}")

# Worst degradation by station bin
print("\nWorst degradation by event size:")
for bin_label in labels_stations:
    leaked = test_burst[(test_burst["station_bin"] == bin_label) & test_burst["is_leaked"]]
    clean = test_burst[(test_burst["station_bin"] == bin_label) & ~test_burst["is_leaked"]]
    if len(leaked) >= 10 and len(clean) >= 10:
        delta = leaked["correct"].mean() - clean["correct"].mean()
        print(f"  {bin_label}: ΔRecall = {delta:+.1%}")

print("\nDone!")
