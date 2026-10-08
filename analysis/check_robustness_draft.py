import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from common import load_split_with_v2

meta_test = load_split_with_v2("test")
meta_tv = pd.concat([pd.read_parquet("data/meta_train.parquet"), pd.read_parquet("data/meta_val.parquet")], ignore_index=True)
tags = pd.read_parquet("data/test_catalog_tags.parquet")

meta_test["start_datetime"] = pd.to_datetime(meta_test.start_datetime)
meta_tv["start_datetime"] = pd.to_datetime(meta_tv.start_datetime)
meta_test["y"] = (meta_test.manual_label != 0).astype(int)
meta_tv["y"] = (meta_tv.manual_label != 0).astype(int)

tv_b = set(meta_tv[meta_tv.y == 1].start_datetime.dt.floor("15min"))
is_b = meta_test.y.values == 1
meta_test["leak"] = is_b & meta_test.start_datetime.dt.floor("15min").isin(tv_b).values
meta_test["cat_any"] = tags.cat_any.values

y = meta_test.y.values
B_L = (y == 1) & meta_test.leak.values
B_C = (y == 1) & ~meta_test.leak.values
B_CAT = (y == 1) & meta_test.cat_any.values

runs = ["published_FlareSense-v2", "purged_seed0", "purged_seed1", "purged_seed2",
        "random_control_seed0", "random_control_seed1", "random_control_seed2"]

TARGET_FPR = 0.0084317
TARGET_REC = 0.7258536

print("=" * 105)
print("1. OPERATING POINT COMPARISON: TEST-MATCHED FPR = 0.843% (IDENTICAL TO PUBLISHED V2)")
print("=" * 105)
print(f"{'Run':<25} | {'Threshold':<9} | {'FPR':<7} | {'Recall':<7} | {'Raw Gap':<10} | {'Cat Gap':<10} | {'Clean Cat Rec':<14} | {'AUROC':<7}")
print("-" * 105)

data_by_run = {}

for r in runs:
    if r == "published_FlareSense-v2":
        logits = meta_test.v2_logit.values
    else:
        t = pd.read_parquet(f"data/retrain/{r}_test.parquet")
        logits = t.logit.values
    data_by_run[r] = logits
    
    neg_logits = np.sort(logits[y == 0])
    thr_fpr = float(np.quantile(neg_logits, 1 - TARGET_FPR))
    
    pred = (logits >= thr_fpr).astype(int)
    rec_all = pred[y == 1].mean() * 100
    actual_fpr = pred[y == 0].mean() * 100
    
    gap_all = (pred[B_L].mean() - pred[B_C].mean()) * 100
    gap_cat = (pred[B_L & B_CAT].mean() - pred[B_C & B_CAT].mean()) * 100
    rec_clean_cat = pred[B_C & B_CAT].mean() * 100
    auroc = roc_auc_score(y, logits)
    
    print(f"{r:<25} | {thr_fpr:9.4f} | {actual_fpr:5.2f}%  | {rec_all:5.2f}%   | {gap_all:6.2f}%    | {gap_cat:6.2f}%    | {rec_clean_cat:6.2f}%        | {auroc:.4f}")

print("\n" + "=" * 105)
print("2. OPERATING POINT COMPARISON: TEST-MATCHED RECALL = 72.59% (IDENTICAL TO PUBLISHED V2)")
print("=" * 105)
print(f"{'Run':<25} | {'Threshold':<9} | {'FPR':<7} | {'Recall':<7} | {'Raw Gap':<10} | {'Cat Gap':<10} | {'Clean Cat Rec':<14}")
print("-" * 105)

for r in runs:
    logits = data_by_run[r]
    pos_logits = np.sort(logits[y == 1])
    thr_rec = float(np.quantile(pos_logits, 1 - TARGET_REC))
    
    pred = (logits >= thr_rec).astype(int)
    rec_all = pred[y == 1].mean() * 100
    actual_fpr = pred[y == 0].mean() * 100
    
    gap_all = (pred[B_L].mean() - pred[B_C].mean()) * 100
    gap_cat = (pred[B_L & B_CAT].mean() - pred[B_C & B_CAT].mean()) * 100
    rec_clean_cat = pred[B_C & B_CAT].mean() * 100
    
    print(f"{r:<25} | {thr_rec:9.4f} | {actual_fpr:5.2f}%  | {rec_all:5.2f}%   | {gap_all:6.2f}%    | {gap_cat:6.2f}%    | {rec_clean_cat:6.2f}%")

print("\n" + "=" * 105)
print("3. DOUBLE-DIFFERENCE (DD = Control - Purged) AT EQUAL OPERATING POINTS")
print("=" * 105)
for s in [0, 1, 2]:
    p_log = data_by_run[f"purged_seed{s}"]
    rc_log = data_by_run[f"random_control_seed{s}"]
    
    # At matched FPR = 0.84%
    p_thr = float(np.quantile(np.sort(p_log[y == 0]), 1 - TARGET_FPR))
    rc_thr = float(np.quantile(np.sort(rc_log[y == 0]), 1 - TARGET_FPR))
    
    p_pred = (p_log >= p_thr).astype(int)
    rc_pred = (rc_log >= rc_thr).astype(int)
    
    dd_raw = (rc_pred[B_L].mean() - rc_pred[B_C].mean()) - (p_pred[B_L].mean() - p_pred[B_C].mean())
    dd_cat = (rc_pred[B_L & B_CAT].mean() - rc_pred[B_C & B_CAT].mean()) - (p_pred[B_L & B_CAT].mean() - p_pred[B_C & B_CAT].mean())
    
    print(f"Seed {s} at matched FPR (0.84%):")
    print(f"  All-burst DD:   {dd_raw*100:+.2f} pp")
    print(f"  Catalog DD:     {dd_cat*100:+.2f} pp")
    print(f"  Recall Control: {rc_pred[y==1].mean()*100:.1f}% | Recall Purged: {p_pred[y==1].mean()*100:.1f}%")

print("\n" + "=" * 105)
print("4. THRESHOLD-FREE AUROC AND AP COMPARISONS")
print("=" * 105)
with open("results/retraining_experiment.json", "r") as f:
    res = json.load(f)

print(f"{'Run':<25} | {'AUC Leak Cat':<13} | {'AUC Clean Cat':<14} | {'AUC Cat Gap':<11} | {'AUC Leak All':<13} | {'AUC Clean All':<14} | {'AUC All Gap':<11}")
print("-" * 105)
for k, v in res["runs"].items():
    alc = v.get("auroc_leaked_vs_neg_catalog", 0)
    acc = v.get("auroc_clean_vs_neg_catalog", 0)
    gapc = v.get("auroc_gap_catalog", 0)
    ala = v.get("auroc_leaked_vs_neg", 0)
    aca = v.get("auroc_clean_vs_neg", 0)
    gapa = v.get("auroc_gap", 0)
    print(f"{k:<25} | {alc:11.4f}   | {acc:12.4f}   | {gapc:9.4f}   | {ala:11.4f}   | {aca:12.4f}   | {gapa:9.4f}")

