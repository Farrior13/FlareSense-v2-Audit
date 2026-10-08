"""Step 05 - label-origin control using the public e-CALLISTO routine burst catalog
(C. Monstein, https://soleil.i4ds.ch/solarradio/data/BurstLists/2010-yyyy_Monstein/).

Motivation: the paper (Sec. 4) states that *test* labels were re-verified by the PI in a
second blind pass, while train/val labels come from the routine catalog. Consequently the
HF test split has 13.4% bursts vs 9.2% in train/val, and the paper reports that the routine
catalog reaches only 63% recall on the clean test labels (Table 3). Test bursts that were
*added* in the second pass (absent from the routine catalog) are likely weak bursts whose
train-set analogues are labelled negative -> a label-noise confound for the leaked-vs-clean
comparison. Here we tag every burst sample as `catalog_listed` (a routine-catalog entry
overlaps its 15-min window for its station, or for the whole network) or `added`.

Writes results/catalog_label_origin.json and data/burst_catalog_parsed.parquet.
Catalog text files are expected in data/catalog/e-CALLISTO_YYYY_MM.txt (download with
analysis/download_catalog.ps1 or the loop in README).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS, cluster_bootstrap, event_context, leak_flags, load_split_with_v2, load_trainval, save_json  # noqa

CAT_DIR = DATA / "catalog"
LINE = re.compile(r"^(\d{8})\s+(\d{2}:\d{2})-(\d{2}:\d{2})\s+(\S+)\s*(.*)$")


def parse_catalog() -> pd.DataFrame:
    rows = []
    for f in sorted(CAT_DIR.glob("e-CALLISTO_*.txt")):
        for raw in f.read_text(encoding="latin-1").splitlines():
            m = LINE.match(raw.strip())
            if not m:
                continue
            d, t0, t1, typ, st = m.groups()
            if "#" in t0 or "#" in t1:
                continue
            try:
                start = pd.Timestamp(f"{d} {t0}")
                end = pd.Timestamp(f"{d} {t1}")
            except (ValueError, pd.errors.ParserError):
                parse_catalog.n_malformed = getattr(parse_catalog, "n_malformed", 0) + 1
                continue
            if end < start:          # crosses midnight
                end += pd.Timedelta(days=1)
            stations = [s.strip().strip("()").strip() for s in st.split(",") if s.strip()]
            rows.append(dict(start=start, end=end + pd.Timedelta(minutes=1), type=typ,
                             stations=stations, generic=(len(stations) == 0 or
                                                         any(s.lower() == "e-callisto" for s in stations))))
    cat = pd.DataFrame(rows)
    return cat


def station_key(antenna: str) -> str:
    """'ALASKA-COHOE_63' -> 'ALASKA-COHOE' (catalog uses station names without focus code)."""
    return re.sub(r"_\d+$", "", antenna).upper()


def tag(df: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """For each row: listed_station (catalog entry overlapping [t, t+15) naming this station),
    listed_any (any catalog entry overlapping the window)."""
    s = cat.start.values.astype("datetime64[s]").astype(np.int64)
    e = cat.end.values.astype("datetime64[s]").astype(np.int64)
    order = np.argsort(s); s, e = s[order], e[order]
    sts = [set(map(str.upper, x)) for x in cat.stations.values[order]]
    gen = cat.generic.values[order]
    emax = np.maximum.accumulate(e)
    t0 = df.start_datetime.values.astype("datetime64[s]").astype(np.int64)
    t1 = t0 + 15 * 60
    keys = df.antenna.map(station_key).values
    any_, st_, gen_ = np.zeros(len(df), bool), np.zeros(len(df), bool), np.zeros(len(df), bool)
    for i in range(len(df)):
        hi = np.searchsorted(s, t1[i], "left")        # entries starting before window end
        lo = np.searchsorted(emax, t0[i], "right")    # first entry that could end after window start
        for j in range(lo, hi):
            if e[j] > t0[i]:
                any_[i] = True
                if keys[i] in sts[j]:
                    st_[i] = True
                if gen[j]:
                    gen_[i] = True
    out = df.copy()
    out["cat_any"] = any_
    out["cat_station"] = st_
    out["cat_generic"] = gen_
    return out


def main():
    cat = parse_catalog()
    cat.assign(stations=cat.stations.map(", ".join)).to_parquet(DATA / "burst_catalog_parsed.parquet", index=False)
    R = {"catalog": {"n_entries": len(cat), "date_range": [str(cat.start.min()), str(cat.end.max())],
                     "share_generic_station": float(cat.generic.mean()),
                     "n_malformed_rows_skipped": getattr(parse_catalog, "n_malformed", 0),
                     "types": cat.type.value_counts().to_dict()}}

    test = load_split_with_v2("test")
    tv = load_trainval()
    flags = leak_flags(test, tv)
    test = pd.concat([test, event_context(test, tv)], axis=1)
    test["leak"] = flags["bucket_15min"]

    # --- validation of the matching on train/val (labels there come from the routine catalog)
    tvt = tag(tv, cat)
    R["matching_validation_trainval"] = {
        "bursts_listed_station": float(tvt[tvt.y == 1].cat_station.mean()),
        "bursts_listed_any": float(tvt[tvt.y == 1].cat_any.mean()),
        "nonbursts_listed_any": float(tvt[tvt.y == 0].cat_any.mean()),
    }
    tt = tag(test, cat)
    b = tt[tt.y == 1].copy()
    nb = tt[tt.y == 0]
    R["test_label_origin"] = {
        "bursts": len(b),
        "bursts_listed_station": float(b.cat_station.mean()),
        "bursts_listed_any": float(b.cat_any.mean()),
        "nonbursts_listed_any": float(nb.cat_any.mean()),
        "by_leak": {
            "leaked_listed_station": float(b[b.leak].cat_station.mean()),
            "clean_listed_station": float(b[~b.leak].cat_station.mean()),
            "leaked_listed_any": float(b[b.leak].cat_any.mean()),
            "clean_listed_any": float(b[~b.leak].cat_any.mean()),
        },
    }
    # --- the key control: leaked vs clean recall WITHIN label-origin strata
    ctrl = {}
    for name, (pc, prc) in MODELS.items():
        d = {}
        for orig_name, mask in [("catalog_listed_any", b.cat_any), ("not_listed_any", ~b.cat_any),
                                ("catalog_listed_station", b.cat_station), ("not_listed_station", ~b.cat_station)]:
            g = b[mask]
            gl, gc = g[g.leak], g[~g.leak]
            d[orig_name] = {"n_leaked": len(gl), "n_clean": len(gc),
                            "recall_leaked": float(gl[pc].mean()) if len(gl) else None,
                            "recall_clean": float(gc[pc].mean()) if len(gc) else None,
                            "gap_pp": float(100 * (gl[pc].mean() - gc[pc].mean())) if len(gl) and len(gc) else None}
        # cluster-bootstrap CI for the gap among catalog-listed (any) bursts
        g = b[b.cat_any].reset_index(drop=True)
        if g.leak.any() and (~g.leak).any():
            ci = cluster_bootstrap(g, lambda i, g=g: g[pc].values[i][g.leak.values[i]].mean()
                                   - g[pc].values[i][~g.leak.values[i]].mean(), B=2000)
            d["catalog_listed_any"]["gap_ci95_pp_cluster"] = [100 * ci[0], 100 * ci[1]]
        ctrl[name] = d
    R["leak_gap_within_label_origin"] = ctrl

    # --- regression among catalog-listed bursts: leak effect net of event size, station, time of day
    import statsmodels.formula.api as smf
    g = b[b.cat_any].copy()
    g["leak_i"] = g.leak.astype(int)
    g["log_n"] = np.log(g.n_st_all.clip(lower=1))
    g["hour_block"] = (g.start_datetime.dt.hour // 6).astype(str)
    reg = {}
    for name, (pc, _) in MODELS.items():
        g["_p"] = g[pc].astype(int)
        fit = smf.logit("_p ~ leak_i + log_n + C(antenna) + C(hour_block)", data=g).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(g.event_bucket)[0]}, maxiter=300)
        ci = fit.conf_int().loc["leak_i"]
        reg[name] = {"OR_leak": float(np.exp(fit.params["leak_i"])), "OR_ci95": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
                     "p": float(fit.pvalues["leak_i"]), "n": int(len(g))}
        # size-matched (2-3 stations) within catalog-listed
        s = g[g.n_st_all.isin([2, 3])]
        reg[name]["size_matched_2to3"] = {"n_leaked": int(s.leak.sum()), "n_clean": int((~s.leak).sum()),
                                          "gap_pp": float(100 * (s[s.leak][pc].mean() - s[~s.leak][pc].mean()))}
    dd = g.v2_pred.astype(int) - g.old_pred.astype(int)
    reg["DiD_v2_minus_old_pp"] = float(100 * (dd[g.leak].mean() - dd[~g.leak].mean()))
    gg = g.reset_index(drop=True)
    ci = cluster_bootstrap(gg, lambda i, t=gg: ((t.v2_pred.values[i] - t.old_pred.values[i])[t.leak.values[i]].mean()
                                                 - (t.v2_pred.values[i] - t.old_pred.values[i])[~t.leak.values[i]].mean()), B=2000)
    reg["DiD_ci95_pp_cluster"] = [100 * ci[0], 100 * ci[1]]
    # recall on bursts added in the PI second pass (not in routine catalog for any station)
    reg["recall_on_PI_added_bursts"] = {n: float(b[~b.cat_any][pc].mean()) for n, (pc, _) in MODELS.items()}
    reg["recall_on_catalog_listed_bursts"] = {n: float(b[b.cat_any][pc].mean()) for n, (pc, _) in MODELS.items()}
    R["catalog_listed_controls"] = reg
    save_json(R, "catalog_label_origin.json")
    tt[["start_datetime", "antenna", "y", "cat_any", "cat_station", "cat_generic"]].to_parquet(
        DATA / "test_catalog_tags.parquet", index=False)


if __name__ == "__main__":
    main()
