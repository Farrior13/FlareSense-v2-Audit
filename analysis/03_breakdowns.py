"""Step 03 - breakdowns: per-station effects, time of day, negative-class difficulty,
FP/FN by station, confidence structure. Writes results/breakdown_results.json."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, event_context, leak_flags, load_split_with_v2, load_trainval, prf, save_json  # noqa: E402

test = load_split_with_v2("test")
tv = load_trainval()
flags = leak_flags(test, tv)
test = pd.concat([test, event_context(test, tv)], axis=1)
test["leak"] = flags["bucket_15min"]
y = test.y.values
R = {}

for name, (pc, prc) in MODELS.items():
    d = {}
    # per-station: full vs clean F1 and leaked vs clean recall
    rows = []
    for a, g in test.groupby("antenna"):
        full = prf(g.y, g[pc]); cl_ = g[~g.leak]; clean = prf(cl_.y, cl_[pc])
        b = g[g.y == 1]
        rows.append({"antenna": a, "n": len(g), "bursts_full": int(g.y.sum()), "bursts_clean": int((b.leak == 0).sum()),
                     "f1_full": full["f1"], "f1_clean": clean["f1"], "dF1": clean["f1"] - full["f1"],
                     "recall_leaked": float(b[b.leak][pc].mean()) if b.leak.any() else None,
                     "recall_clean": float(b[~b.leak][pc].mean()) if (~b.leak).any() else None,
                     "fp": full["fp"], "negatives": int((g.y == 0).sum()), "fp_rate": full["fpr"],
                     "fn": full["fn"], "miss_rate": 1 - full["recall"]})
    d["per_station"] = sorted(rows, key=lambda r: r["dF1"] if r["dF1"] == r["dF1"] else 0)
    # time of day (UTC, 6h blocks), bursts only
    tb = test[test.y == 1]
    tod = []
    for h in range(4):
        g = tb[(tb.start_datetime.dt.hour // 6) == h]
        tod.append({"utc": f"{6*h:02d}-{6*h+6:02d}", "n_leaked": int(g.leak.sum()), "n_clean": int((~g.leak).sum()),
                    "recall_leaked": float(g[g.leak][pc].mean()), "recall_clean": float(g[~g.leak][pc].mean())})
    d["time_of_day"] = tod
    # negative class difficulty
    neg = test[test.y == 0][prc].values
    edges = [0, .01, .05, .10, .5, 1.0001]
    hist = np.histogram(neg, bins=edges)[0] / len(neg)
    d["negative_prob_distribution"] = {"bins": ["<0.01", "0.01-0.05", "0.05-0.10", "0.10-0.50", ">=0.50"],
                                       "share": hist.tolist(), "median": float(np.median(neg)),
                                       "share_below_0.05": float((neg < .05).mean())}
    # confidence structure among bursts
    pb = tb[prc].values; lk = tb.leak.values
    mid = (pb >= .3) & (pb < .5)
    d["burst_confidence"] = {
        "quantiles_leaked": {q: float(np.quantile(pb[lk], q)) for q in (.1, .25, .5, .75)},
        "quantiles_clean": {q: float(np.quantile(pb[~lk], q)) for q in (.1, .25, .5, .75)},
        "share_clean_among_prob_0.3_0.5": float((~lk[mid]).mean()) if mid.any() else None,
        "share_clean_among_all_bursts": float((~lk).mean())}
    # clean bursts by number of observing stations (structure of the clean subset)
    d["clean_by_n_stations"] = [{"n_stations_all": k, "n_leaked": int(g.leak.sum()), "n_clean": int((~g.leak).sum())}
                                for k, g in tb.groupby(np.clip(tb.n_st_all, 1, 11))]
    R[name] = d

save_json(R, "breakdown_results.json")
