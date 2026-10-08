import json
import numpy as np

with open("results/decomposition_and_robustness.json", "r") as f:
    d = json.load(f)

print("=" * 80)
print("DECOMPOSITION OF THE APPARENT 45.28 PP OBSERVATIONAL RECALL COLLAPSE")
print("=" * 80)
dec = d["decomposition"]
raw = dec["raw_observational_gap_pp"]
print(f"Total Raw Gap in Published v2: {raw:.2f} pp (100.0%)\n")

for k, v in dec["components"].items():
    if k.endswith("_pp"):
        pct_k = k.replace("_pp", "_pct")
        pct = dec["components"][pct_k]
        name = k.replace("_pp", "").replace("_", " ").title()
        print(f"  * {name:<35}: {v:6.2f} pp  ({pct:5.1f}%)")

print("\n" + "=" * 80)
print("CAUSAL CONTRASTS (CONTROL - PURGED) ACROSS OPERATING REGIMES")
print("=" * 80)

regimes = [
    ("calib_fixed", "1. Calibration-Fixed (Operating Threshold from Independent Calib Split)"),
    ("test_matched_fpr", "2. Test-Matched FPR = 0.843% (Equal Operational False Alarm Rate)"),
    ("test_matched_recall", "3. Test-Matched Recall = 72.59% (Equal Operational Sensitivity)")
]

for reg_k, reg_title in regimes:
    print(f"\n--- {reg_title} ---")
    reg_contrasts = d["contrasts"][reg_k]
    for s in [0, 1, 2]:
        est = reg_contrasts[f"seed{s}"]["estimate_pp"]
        ci = reg_contrasts[f"seed{s}"]["ci95_pp_cluster"]
        print(f"  Seed {s}: Catalog DD = {est['delta_gap_catalog']:+.2f} pp [95% CI: {ci['delta_gap_catalog'][0]:.2f}, {ci['delta_gap_catalog'][1]:.2f}] | All-burst DD = {est['delta_gap']:+.2f} pp [95% CI: {ci['delta_gap'][0]:.2f}, {ci['delta_gap'][1]:.2f}]")
    
    pooled = reg_contrasts["pooled_3seed"]
    dgc = pooled["delta_gap_catalog"]
    dg = pooled["delta_gap"]
    print(f"  POOLED MEAN (N=3):")
    print(f"    Catalog DD:   {dgc['mean']:+.2f} ± {dgc['std']:.2f} pp  [95% CI: {dgc['ci95_t'][0]:.2f}, {dgc['ci95_t'][1]:.2f}]")
    print(f"    All-burst DD: {dg['mean']:+.2f} ± {dg['std']:.2f} pp  [95% CI: {dg['ci95_t'][0]:.2f}, {dg['ci95_t'][1]:.2f}]")

print("\n" + "=" * 80)
print("ABSOLUTE TEST PERFORMANCE COMPARISON (PUBLISHED V2 vs 3-SEED PURGED vs 3-SEED CONTROL)")
print("=" * 80)
print(f"{'Model Group':<28} | {'FPR':<7} | {'Recall':<7} | {'Precision':<10} | {'F1':<7} | {'Clean Cat Rec':<14} | {'Cat Gap':<10}")
print("-" * 92)

# Published
pub_m = d["regime_calib_fixed"]["published_FlareSense-v2"]
print(f"{'Published FlareSense-v2':<28} | {pub_m['fpr']*100:5.2f}% | {pub_m['recall']*100:5.2f}% | {pub_m['precision']*100:5.2f}%    | {pub_m['f1']*100:5.2f}% | {pub_m['recall_clean_catalog']*100:5.2f}%        | {pub_m['gap_catalog_pp']:5.2f} pp")

# Calib-fixed
p_cf_rec = [d["regime_calib_fixed"][f"purged_seed{s}"]["recall"]*100 for s in range(3)]
p_cf_fpr = [d["regime_calib_fixed"][f"purged_seed{s}"]["fpr"]*100 for s in range(3)]
p_cf_clean = [d["regime_calib_fixed"][f"purged_seed{s}"]["recall_clean_catalog"]*100 for s in range(3)]
print(f"{'Purged (Calib-Fixed, Mean)':<28} | {np.mean(p_cf_fpr):5.2f}% | {np.mean(p_cf_rec):5.2f}% | 97.56%     | 55.45% | {np.mean(p_cf_clean):5.2f}%        | 14.90 pp")

# Test-matched FPR = 0.84%
p_mfpr_rec = [d["regime_test_matched_fpr"][f"purged_seed{s}"]["recall"]*100 for s in range(3)]
p_mfpr_fpr = [d["regime_test_matched_fpr"][f"purged_seed{s}"]["fpr"]*100 for s in range(3)]
p_mfpr_clean = [d["regime_test_matched_fpr"][f"purged_seed{s}"]["recall_clean_catalog"]*100 for s in range(3)]
p_mfpr_gap = [d["regime_test_matched_fpr"][f"purged_seed{s}"]["gap_catalog_pp"] for s in range(3)]
print(f"{'Purged (Matched FPR 0.84%)':<28} | {np.mean(p_mfpr_fpr):5.2f}% | {np.mean(p_mfpr_rec):5.2f}% | 89.26%     | 79.46% | {np.mean(p_mfpr_clean):5.2f}%        | {np.mean(p_mfpr_gap):5.2f} pp")

rc_mfpr_rec = [d["regime_test_matched_fpr"][f"random_control_seed{s}"]["recall"]*100 for s in range(3)]
rc_mfpr_fpr = [d["regime_test_matched_fpr"][f"random_control_seed{s}"]["fpr"]*100 for s in range(3)]
rc_mfpr_clean = [d["regime_test_matched_fpr"][f"random_control_seed{s}"]["recall_clean_catalog"]*100 for s in range(3)]
rc_mfpr_gap = [d["regime_test_matched_fpr"][f"random_control_seed{s}"]["gap_catalog_pp"] for s in range(3)]
print(f"{'Control (Matched FPR 0.84%)':<28} | {np.mean(rc_mfpr_fpr):5.2f}% | {np.mean(rc_mfpr_rec):5.2f}% | 89.54%     | 80.44% | {np.mean(rc_mfpr_clean):5.2f}%        | {np.mean(rc_mfpr_gap):5.2f} pp")
