"""Step 06 - evaluate the controlled retraining experiment (train/train_leakage_experiment.py).

For every available run data/retrain/{arm}_seed{s}_test.parquet (+ _calib.parquet):
  * threshold-free: AUROC / AP on the full test set; AUROC of leaked-bursts-vs-negatives and
    clean-bursts-vs-negatives (same negatives) -> leakage gap without any threshold
  * operating point fixed on the CALIBRATION set (never on test): FPR_calib = FPR of the
    published model at the paper's operating point (0.84%)
  * recall for leaked / clean bursts, overall and within catalog-listed bursts (label-origin control)
  * key contrast: gap(random_control) - gap(purged), event-cluster bootstrap, paired on test samples
The published FlareSense-v2 is included as a reference row.
Writes results/retraining_experiment.json.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, cluster_bootstrap, leak_flags, load_split_with_v2, load_trainval, prf, save_json  # noqa

RUN_DIR = DATA / "retrain"
test = load_split_with_v2("test")
tv = load_trainval()
test["leak"] = leak_flags(test, tv)["bucket_15min"]
tags = pd.read_parquet(DATA / "test_catalog_tags.parquet")      # from 05_catalog_label_origin.py
assert (tags.start_datetime.values == test.start_datetime.values).all()
test["cat_any"] = tags.cat_any.values
y = test.y.values
neg = y == 0
B_L = (y == 1) & test.leak.values
B_C = (y == 1) & ~test.leak.values
B_CAT = (y == 1) & test.cat_any.values
FPR_TARGET = prf(y, test.v2_pred)["fpr"]


def auc_vs_neg(score, pos_mask):
    m = pos_mask | neg
    return float(roc_auc_score(y[m], score[m]))


def summarize(score, thr):
    pred = (score >= thr).astype(int)
    r = {"auroc": float(roc_auc_score(y, score)), "ap": float(average_precision_score(y, score)),
         "auroc_leaked_vs_neg": auc_vs_neg(score, B_L), "auroc_clean_vs_neg": auc_vs_neg(score, B_C),
         "auroc_leaked_vs_neg_catalog": auc_vs_neg(score, B_L & B_CAT),
         "auroc_clean_vs_neg_catalog": auc_vs_neg(score, B_C & B_CAT),
         "at_threshold": prf(y, pred),
         "recall_leaked": float(pred[B_L].mean()), "recall_clean": float(pred[B_C].mean()),
         "recall_leaked_catalog": float(pred[B_L & B_CAT].mean()), "recall_clean_catalog": float(pred[B_C & B_CAT].mean())}
    r["gap_pp"] = 100 * (r["recall_leaked"] - r["recall_clean"])
    r["gap_catalog_pp"] = 100 * (r["recall_leaked_catalog"] - r["recall_clean_catalog"])
    r["auroc_gap"] = r["auroc_leaked_vs_neg"] - r["auroc_clean_vs_neg"]
    r["auroc_gap_catalog"] = r["auroc_leaked_vs_neg_catalog"] - r["auroc_clean_vs_neg_catalog"]
    return r, pred


R = {"fpr_target_from_published_model": FPR_TARGET, "runs": {}}
r, _ = summarize(test.v2_logit.values, np.log(0.426 / 0.574) * 0.4974)
R["runs"]["published_FlareSense-v2"] = r

preds = {}
for f in sorted(RUN_DIR.glob("*_test.parquet")):
    tag = f.name[:-len("_test.parquet")]
    if tag.endswith("_smoke"):
        continue
    t = pd.read_parquet(f)
    assert len(t) == len(test) and (t.hf_index.values == np.arange(len(test))).all()
    c = pd.read_parquet(RUN_DIR / f"{tag}_calib.parquet")
    cn = np.sort(c[c.y == 0].logit.values)
    thr = float(np.quantile(cn, 1 - FPR_TARGET))           # FPR on calibration negatives = target
    r, pred = summarize(t.logit.values, thr)
    r["threshold_logit"] = thr
    r["calib_fpr"] = float((cn >= thr).mean())
    R["runs"][tag] = r
    preds[tag] = pred

# paired contrasts random_control vs purged (same seed)
contr = {}
for tag in preds:
    if not tag.startswith("purged_"):
        continue
    ctrl = tag.replace("purged_", "random_control_")
    if ctrl not in preds:
        continue
    pp, pc = preds[tag], preds[ctrl]

    def stat(i, pp=pp, pc=pc):
        bl, bc, cat = B_L[i], B_C[i], B_CAT[i]
        g_c = pc[i][bl].mean() - pc[i][bc].mean(); g_p = pp[i][bl].mean() - pp[i][bc].mean()
        g_cc = pc[i][bl & cat].mean() - pc[i][bc & cat].mean(); g_pc = pp[i][bl & cat].mean() - pp[i][bc & cat].mean()
        return {"delta_gap": g_c - g_p, "delta_gap_catalog": g_cc - g_pc,
                "delta_recall_leaked": pc[i][bl].mean() - pp[i][bl].mean(),
                "delta_recall_clean": pc[i][bc].mean() - pp[i][bc].mean()}

    est = stat(np.arange(len(test)))
    contr[f"{ctrl}_minus_{tag}"] = {"estimate_pp": {k: 100 * v for k, v in est.items()},
                                    "ci95_pp_cluster": {k: [100 * a, 100 * b] for k, (a, b) in
                                                        cluster_bootstrap(test, stat, B=2000).items()}}
R["contrasts"] = contr
save_json(R, "retraining_experiment.json")
