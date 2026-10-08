import json
import numpy as np

with open("results/retraining_experiment.json", "r") as f:
    d = json.load(f)

print("=" * 80)
print("FLARESENSE-V2 CONTROLLED RETRAINING AUDIT: FULL 3-SEED RESULTS")
print("=" * 80)

print("\n--- INDIVIDUAL RUN METRICS ---")
print(f"{'Run Tag':<26} | {'AUROC':<7} | {'AP':<7} | {'Gap (all)':<10} | {'Gap (cat)':<10} | {'Rec Clean Cat':<14} | {'Rec Leaked Cat':<14}")
print("-" * 105)

for k, v in d["runs"].items():
    auroc = v.get("auroc", 0)
    ap = v.get("ap", 0)
    gap = v.get("gap_pp", 0)
    cat_gap = v.get("gap_catalog_pp", 0)
    rec_clean_cat = v.get("recall_clean_catalog", 0) * 100
    rec_leaked_cat = v.get("recall_leaked_catalog", 0) * 100
    print(f"{k:<26} | {auroc:.4f}  | {ap:.4f}  | {gap:6.2f}%    | {cat_gap:6.2f}%    | {rec_clean_cat:6.2f}%        | {rec_leaked_cat:6.2f}%")

print("\n--- CAUSAL CONTRASTS (Control - Purged: Double-Difference) ---")
deltas_gap = []
deltas_cat_gap = []

for k, v in d["contrasts"].items():
    est = v["estimate_pp"]
    ci = v["ci95_pp_cluster"]
    dg = est["delta_gap"]
    dcg = est["delta_gap_catalog"]
    deltas_gap.append(dg)
    deltas_cat_gap.append(dcg)
    print(f"\n{k}:")
    print(f"  All-burst gap (DD):   {dg:+.2f} pp [95% CI: {ci['delta_gap'][0]:.2f}, {ci['delta_gap'][1]:.2f}]")
    print(f"  Catalog gap (DD):     {dcg:+.2f} pp [95% CI: {ci['delta_gap_catalog'][0]:.2f}, {ci['delta_gap_catalog'][1]:.2f}]")
    print(f"  Delta recall leaked:  {est['delta_recall_leaked']:+.2f} pp [95% CI: {ci['delta_recall_leaked'][0]:.2f}, {ci['delta_recall_leaked'][1]:.2f}]")
    print(f"  Delta recall clean:   {est['delta_recall_clean']:+.2f} pp [95% CI: {ci['delta_recall_clean'][0]:.2f}, {ci['delta_recall_clean'][1]:.2f}]")

print("\n" + "=" * 80)
print(f"MULTI-SEED AGGREGATION (N = 3 SEEDS):")
print(f"  Mean Catalog Gap DD:   {np.mean(deltas_cat_gap):+.2f} pp ± {np.std(deltas_cat_gap, ddof=1):.2f} pp")
print(f"  Mean All-burst Gap DD: {np.mean(deltas_gap):+.2f} pp ± {np.std(deltas_gap, ddof=1):.2f} pp")
print("=" * 80)
