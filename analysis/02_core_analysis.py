"""Step 02 - core analysis: paper reproduction, event overlap, full-vs-clean evaluation,
confidence shift, confound controls. Writes results/core_results.json.

All analyses are run for FlareSense-v2 (primary) and for the older HF-provided
predictions (reference model that was NOT trained on this split's train set).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MODELS, THR_CAL, cluster_bootstrap, event_context, leak_flags, load_split_with_v2,  # noqa: E402
                    load_trainval, macro_prf, prevalence_adjusted_precision, prf, save_json)

B = 2000
R = {}
test = load_split_with_v2("test")
tv = load_trainval()
flags = leak_flags(test, tv)
ctx = event_context(test, tv)
test = pd.concat([test, ctx], axis=1)
y = test.y.values
is_b = y == 1

# ------------------------------------------------------------------ 1. dataset
R["dataset"] = {
    "n_trainval": len(tv), "n_test": len(test),
    "test_bursts": int(is_b.sum()), "test_nonbursts": int((~is_b).sum()),
    "trainval_bursts": int(tv.y.sum()),
    "test_date_range": [str(test.start_datetime.min()), str(test.start_datetime.max())],
    "n_antennas_test": int(test.antenna.nunique()),
    "manual_label_values_test": {str(k): int(v) for k, v in test.manual_label.value_counts().items()},
}

# ------------------------------------------------------------------ 2. paper reproduction
rep = {}
for name, (pc, _) in MODELS.items():
    rep[name] = {"micro": prf(y, test[pc]), "macro_per_antenna": macro_prf(test, pc)}
rep["FlareSense-v2@logit>0"] = {"micro": prf(y, test.v2_pred_05), "macro_per_antenna": macro_prf(test, "v2_pred_05")}
rep["paper_table3"] = {"precision": 0.93, "recall": 0.7315, "threshold_calibrated": THR_CAL}
rep["agreement_v2_vs_old"] = {
    "pearson_logits": float(np.corrcoef(test.logits, test.v2_logit)[0, 1]),
    "label_agreement": float((test.v2_pred_05 == test.old_pred).mean()),
}
R["paper_reproduction"] = rep

# ------------------------------------------------------------------ 3. overlap statistics
ov = {}
for k, f in flags.items():
    ov[k] = {"leaked": int(f.sum()), "clean": int((is_b & ~f).sum()), "pct_leaked": float(f.sum() / is_b.sum())}
tb = test[is_b]
leak15 = flags["bucket_15min"][is_b]
ev = tb.groupby("event_bucket").agg(n_test_st=("antenna", "nunique"))
ov["test_stations_per_event_15min"] = {
    "1": int((ev.n_test_st == 1).sum()), "2": int((ev.n_test_st == 2).sum()),
    "3+": int((ev.n_test_st >= 3).sum()), "5+": int((ev.n_test_st >= 5).sum()), "n_events": int(len(ev))}
ov["median_trainval_stations_per_leaked_sample"] = float(np.median(tb.n_st_tv[leak15]))
ov["clean_subset_coverage"] = {
    "stations": int(tb[~leak15].antenna.nunique()), "of": int(tb.antenna.nunique()),
    "date_range": [str(tb[~leak15].start_datetime.min()), str(tb[~leak15].start_datetime.max())]}
R["overlap"] = ov

# ------------------------------------------------------------------ 4. full vs clean (+ cluster bootstrap)
fvc = {}
for name, (pc, prc) in MODELS.items():
    p = test[pc].values
    prob = test[prc].values
    m = {}
    full = prf(y, p)
    for k, f in flags.items():
        keep = ~f
        cl = prf(y[keep], p[keep])
        n_pos_c = int((y[keep] == 1).sum()); n_neg = int((y == 0).sum())
        bl, bc = f, is_b & ~f
        m[k] = {
            "clean": cl,
            "clean_precision_at_full_prevalence": prevalence_adjusted_precision(
                cl["tp"], cl["fp"], n_pos_c, n_neg, target_prev=is_b.mean()),
            "recall_leaked": float(p[bl].mean()), "recall_clean": float(p[bc].mean()),
            "recall_gap_pp": float(100 * (p[bl].mean() - p[bc].mean())),
            "prob_leaked": {q: float(np.quantile(prob[bl], qq)) for q, qq in
                            [("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("mean", None)] if qq} |
                           {"mean": float(prob[bl].mean())},
            "prob_clean": {q: float(np.quantile(prob[bc], qq)) for q, qq in
                           [("p10", .1), ("p25", .25), ("median", .5), ("p75", .75)]} |
                          {"mean": float(prob[bc].mean())},
        }

    f15 = flags["bucket_15min"]

    def stat(idx, p=p, f15=f15):
        yy, pp, ff = y[idx], p[idx], f15[idx]
        a = prf(yy, pp); c = prf(yy[~ff], pp[~ff])
        b_ = yy == 1
        return {"dF1": c["f1"] - a["f1"], "dRecall": c["recall"] - a["recall"],
                "dPrecision": c["precision"] - a["precision"],
                "recall_gap": pp[ff].mean() - pp[b_ & ~ff].mean(),
                "full_f1": a["f1"], "clean_f1": c["f1"]}

    m["cluster_bootstrap_15min"] = cluster_bootstrap(test, stat, B=B)
    m["full"] = full
    fvc[name] = m
R["full_vs_clean"] = fvc

# ------------------------------------------------------------------ 5. confound controls
cc = {}
tb = test[is_b].copy()
tb["leak"] = flags["bucket_15min"][is_b].astype(int)
tb["log_n_st_all"] = np.log(tb.n_st_all)
tb["hour_block"] = (tb.start_datetime.dt.hour // 6).astype(str)
for name, (pc, prc) in MODELS.items():
    d = {}
    # (a) stratified by total number of observing stations
    strata = []
    for k, g in tb.groupby(np.clip(tb.n_st_all, 1, 6)):
        gl, gc = g[g.leak == 1], g[g.leak == 0]
        strata.append({"n_stations_all": int(k) if k < 6 else "6+", "n_leaked": len(gl), "n_clean": len(gc),
                       "recall_leaked": float(gl[pc].mean()) if len(gl) else None,
                       "recall_clean": float(gc[pc].mean()) if len(gc) else None})
    d["stratified_by_event_size"] = strata
    # (b) size-matched pooled 2-3 stations, cluster bootstrap
    s = tb[tb.n_st_all.isin([2, 3])].reset_index(drop=True)
    gap = s[s.leak == 1][pc].mean() - s[s.leak == 0][pc].mean()
    ci = cluster_bootstrap(s.assign(y=1), lambda i, s=s: s.iloc[i][s.iloc[i].leak == 1][pc].mean()
                           - s.iloc[i][s.iloc[i].leak == 0][pc].mean(), B=B)
    d["size_matched_2to3_stations"] = {"n_leaked": int((s.leak == 1).sum()), "n_clean": int((s.leak == 0).sum()),
                                       "recall_gap_pp": 100 * gap, "ci95_pp_cluster": [100 * ci[0], 100 * ci[1]]}
    # (c) logistic regression with station + time-of-day FE and event-size covariate,
    #     cluster-robust SE by event bucket
    tb["_pred"] = tb[pc].astype(int)
    groups = pd.factorize(tb.event_bucket)[0]
    fit = smf.logit("_pred ~ leak + log_n_st_all + C(antenna) + C(hour_block)", data=tb).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": groups}, maxiter=200)
    ci_ = fit.conf_int().loc["leak"]
    d["logit_FE_cluster"] = {"OR_leak": float(np.exp(fit.params["leak"])),
                             "OR_leak_ci95": [float(np.exp(ci_[0])), float(np.exp(ci_[1]))],
                             "p_leak": float(fit.pvalues["leak"]),
                             "OR_per_log_station": float(np.exp(fit.params["log_n_st_all"])),
                             "n": int(len(tb))}
    # (d) restricted to events seen by <=3 stations in total (where clean exists)
    small = tb[tb.n_st_all <= 3]
    fit2 = smf.logit("_pred ~ leak + log_n_st_all + C(antenna) + C(hour_block)", data=small).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(small.event_bucket)[0]}, maxiter=200)
    ci2 = fit2.conf_int().loc["leak"]
    d["logit_FE_cluster_events_le3_stations"] = {"OR_leak": float(np.exp(fit2.params["leak"])),
                                                 "OR_leak_ci95": [float(np.exp(ci2[0])), float(np.exp(ci2[1]))],
                                                 "p_leak": float(fit2.pvalues["leak"]), "n": int(len(small))}
    # (e) exposure gradient: recall vs number of train/val stations that saw the same event
    eg = []
    for lab, (a, b_) in {"0": (0, 0), "1-2": (1, 2), "3-5": (3, 5), "6-10": (6, 10), "11+": (11, 10 ** 6)}.items():
        g = tb[(tb.n_st_tv >= a) & (tb.n_st_tv <= b_)]
        eg.append({"trainval_stations": lab, "n": len(g), "recall": float(g[pc].mean()),
                   "mean_prob": float(g[prc].mean()), "median_prob": float(g[prc].median())})
    d["exposure_gradient"] = eg
    cc[name] = d
# (f) difference-in-differences between the two models on identical samples
dd = tb.v2_pred.astype(int) - tb.old_pred.astype(int)
cc["DiD_v2_minus_old_leaked_minus_clean_pp"] = float(100 * (dd[tb.leak == 1].mean() - dd[tb.leak == 0].mean()))
ci = cluster_bootstrap(tb.reset_index(drop=True), lambda i, t=tb.reset_index(drop=True): (
    (t.v2_pred.values[i] - t.old_pred.values[i])[t.leak.values[i] == 1].mean()
    - (t.v2_pred.values[i] - t.old_pred.values[i])[t.leak.values[i] == 0].mean()), B=B)
cc["DiD_ci95_pp_cluster"] = [100 * ci[0], 100 * ci[1]]
R["confound_controls"] = cc

save_json(R, "core_results.json")
