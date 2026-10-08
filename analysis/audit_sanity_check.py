import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

print("=" * 80)
print("DEEP AUDIT AND SANITY VERIFICATION OF RETRAINING EXPERIMENT")
print("=" * 80)

# Load test metadata and catalog tags
meta_test = pd.read_parquet("data/meta_test.parquet")
meta_tv = pd.concat([pd.read_parquet("data/meta_train.parquet"), pd.read_parquet("data/meta_val.parquet")], ignore_index=True)
tags = pd.read_parquet("data/test_catalog_tags.parquet")

meta_test["start_datetime"] = pd.to_datetime(meta_test.start_datetime)
meta_tv["start_datetime"] = pd.to_datetime(meta_tv.start_datetime)
meta_test["y"] = (meta_test.manual_label != 0).astype(int)
meta_tv["y"] = (meta_tv.manual_label != 0).astype(int)

# 15-minute event leakage definition
tv_b = set(meta_tv[meta_tv.y == 1].start_datetime.dt.floor("15min"))
is_b = meta_test.y.values == 1
meta_test["leak"] = is_b & meta_test.start_datetime.dt.floor("15min").isin(tv_b).values
meta_test["cat_any"] = tags.cat_any.values

y_test = meta_test.y.values
neg_mask = y_test == 0
B_L = (y_test == 1) & meta_test.leak.values
B_C = (y_test == 1) & ~meta_test.leak.values
B_CAT = (y_test == 1) & meta_test.cat_any.values

print(f"\n[Test Set Verification]")
print(f"  Total test samples: {len(meta_test):,}")
print(f"  Positives: {y_test.sum():,} | Negatives: {(~neg_mask).sum():,}")
print(f"  Leaked bursts: {B_L.sum():,} ({B_L.sum()/y_test.sum()*100:.1f}%)")
print(f"  Clean bursts:  {B_C.sum():,} ({B_C.sum()/y_test.sum()*100:.1f}%)")
print(f"  Catalog-listed bursts: {B_CAT.sum():,} ({B_CAT.sum()/y_test.sum()*100:.1f}%)")
print(f"  Catalog Leaked: {(B_L & B_CAT).sum():,} | Catalog Clean: {(B_C & B_CAT).sum():,}")

# Load experiment results JSON
with open("results/retraining_experiment.json", "r") as f:
    json_data = json.load(f)

print(f"\n[Check Degeneracy / Confusion Matrices across All 6 Retrained Models]")
runs = ["purged_seed0", "purged_seed1", "purged_seed2",
        "random_control_seed0", "random_control_seed1", "random_control_seed2"]

target_fpr = json_data["fpr_target_from_published_model"]
print(f"Target FPR from published model: {target_fpr*100:.3f}%\n")

audit_records = []
for run in runs:
    df_test = pd.read_parquet(f"data/retrain/{run}_test.parquet")
    df_calib = pd.read_parquet(f"data/retrain/{run}_calib.parquet")
    
    # Check shape
    assert len(df_test) == len(meta_test), f"Test size mismatch in {run}"
    assert (df_test.y.values == y_test).all(), f"Label mismatch in {run}"
    
    # Calibration threshold verification
    calib_neg_logits = np.sort(df_calib[df_calib.y == 0].logit.values)
    thr = float(np.quantile(calib_neg_logits, 1 - target_fpr))
    
    # Verify threshold in JSON matches independently computed threshold
    json_thr = json_data["runs"][run]["threshold_logit"]
    np.testing.assert_allclose(thr, json_thr, rtol=1e-5, err_msg=f"Threshold mismatch in {run}")
    
    # Compute test predictions from continuous logits
    logits = df_test.logit.values
    preds = (logits >= thr).astype(int)
    
    tp = int(((y_test == 1) & (preds == 1)).sum())
    fp = int(((y_test == 0) & (preds == 1)).sum())
    fn = int(((y_test == 1) & (preds == 0)).sum())
    tn = int(((y_test == 0) & (preds == 0)).sum())
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    actual_test_fpr = fp / (fp + tn)
    
    # Degeneracy checks
    assert tn > 25000, f"DEGENERATE MODEL DETECTED: TN={tn} in {run}!"
    assert tp > 500, f"DEGENERATE MODEL DETECTED: TP={tp} in {run}!"
    assert fp < 500, f"HIGH FALSE POSITIVE DETECTED: FP={fp} in {run}!"
    
    # Recall breakdowns
    rec_leaked = preds[B_L].mean()
    rec_clean = preds[B_C].mean()
    rec_leaked_cat = preds[B_L & B_CAT].mean()
    rec_clean_cat = preds[B_C & B_CAT].mean()
    
    raw_gap = (rec_leaked - rec_clean) * 100
    cat_gap = (rec_leaked_cat - rec_clean_cat) * 100
    
    auroc = roc_auc_score(y_test, logits)
    ap = average_precision_score(y_test, logits)
    
    audit_records.append({
        "run": run,
        "thr": thr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "prec": prec * 100, "rec": rec * 100, "f1": f1 * 100,
        "fpr": actual_test_fpr * 100,
        "auroc": auroc, "ap": ap,
        "raw_gap": raw_gap, "cat_gap": cat_gap,
        "rec_clean_cat": rec_clean_cat * 100,
        "rec_leaked_cat": rec_leaked_cat * 100
    })

df_audit = pd.DataFrame(audit_records)
print(df_audit[["run", "tp", "fp", "fn", "tn", "prec", "rec", "f1", "fpr"]].to_string(index=False))

print("\n" + "=" * 80)
print("[Independent Verification of Causal Contrasts]")
for s in [0, 1, 2]:
    p = df_audit[df_audit.run == f"purged_seed{s}"].iloc[0]
    rc = df_audit[df_audit.run == f"random_control_seed{s}"].iloc[0]
    
    dd_raw = rc.raw_gap - p.raw_gap
    dd_cat = rc.cat_gap - p.cat_gap
    
    contrast_key = f"random_control_seed{s}_minus_purged_seed{s}"
    json_dd_raw = json_data["contrasts"][contrast_key]["estimate_pp"]["delta_gap"]
    json_dd_cat = json_data["contrasts"][contrast_key]["estimate_pp"]["delta_gap_catalog"]
    ci_cat = json_data["contrasts"][contrast_key]["ci95_pp_cluster"]["delta_gap_catalog"]
    
    np.testing.assert_allclose(dd_raw, json_dd_raw, rtol=1e-5)
    np.testing.assert_allclose(dd_cat, json_dd_cat, rtol=1e-5)
    
    print(f"Seed {s}:")
    print(f"  Catalog Gap Control: {rc.cat_gap:.2f}% | Purged: {p.cat_gap:.2f}%")
    print(f"  Double-Difference Catalog: {dd_cat:+.2f} pp [95% CI: {ci_cat[0]:.2f}, {ci_cat[1]:.2f}]")
    print(f"  Verification: EXACT MATCH with JSON!")
    assert dd_cat > 0, f"Contrast is not positive in seed {s}!"
    assert ci_cat[0] > 0, f"Confidence interval crosses zero in seed {s}!"

print("\n" + "=" * 80)
print("AUDIT RESULT: ALL ASSERTIONS PASSED WITH ZERO DISCREPANCIES!")
print("=" * 80)
