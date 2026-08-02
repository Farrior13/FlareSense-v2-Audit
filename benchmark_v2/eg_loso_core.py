"""
EG-LOSO (Event Graph Leave-One-Station-Out) evaluation core.

This module implements the EG-LOSO benchmark protocol:
  1. For each target station, remove it from the training set.
  2. Build a graph (or temporal baseline) from the remaining data.
  3. Measure how well the global events cover the held-out station's
     local burst observations using IoL (Event Coverage) and IoU.

All graph construction and temporal metrics are imported from
``graph_utils`` — the single source of truth.
"""

import argparse
import pandas as pd
import numpy as np
import networkx as nx
from datasets import load_dataset
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import json

from graph_utils import (
    build_graph,
    compute_temporal_iou,
    compute_event_coverage,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
REPORTS_DIR = Path("event_graph/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config(config_path):
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


# ---------------------------------------------------------------------------
# Event Extraction
# ---------------------------------------------------------------------------

def extract_station_events(df, indices, window_minutes=15):
    """Extract contiguous burst events for a single station.

    Merges consecutive burst observations that **strictly overlap**
    (i.e. next burst starts before the current event ends).  This is
    the same overlap criterion used by ``build_graph``:
    ``start_j < end_i``.

    .. note::
       Pre-fix code added an extra ``window_minutes`` gap tolerance,
       causing systematic over-merging.  The current implementation
       uses strict overlap only.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (needs ``start_datetime``, ``end_datetime``,
        ``manual_label``).
    indices : array-like
        Row indices belonging to the target station's test set.
    window_minutes : int
        Not used for merging (kept for API compatibility); window is
        already baked into ``end_datetime``.

    Returns
    -------
    list[dict]
        Each dict has keys ``start``, ``end``, ``nodes``.
    """
    sub_df = df.loc[indices].copy()
    bursts = sub_df[sub_df["manual_label"] == 1].sort_values("start_datetime")
    if len(bursts) == 0:
        return []

    events = []
    current_start = bursts.iloc[0]["start_datetime"]
    current_end = bursts.iloc[0]["end_datetime"]
    current_nodes = [bursts.index[0]]

    for i in range(1, len(bursts)):
        row = bursts.iloc[i]
        # Strict overlap: same criterion as graph edge creation
        # (start_j < end_i  ⟹  windows overlap)
        if row["start_datetime"] < current_end:
            current_end = max(current_end, row["end_datetime"])
            current_nodes.append(bursts.index[i])
        else:
            events.append({
                "start": current_start,
                "end": current_end,
                "nodes": current_nodes,
            })
            current_start = row["start_datetime"]
            current_end = row["end_datetime"]
            current_nodes = [bursts.index[i]]

    events.append({
        "start": current_start,
        "end": current_end,
        "nodes": current_nodes,
    })
    return events


def extract_global_events(df, components):
    """Convert connected components into global event descriptors.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``start_datetime``, ``end_datetime``, ``manual_label``.
    components : list[set]
        Connected components from the graph.

    Returns
    -------
    list[dict]
        Each dict has keys ``comp_id``, ``start``, ``end``, ``nodes``,
        ``label`` (1 if any burst node is present).
    """
    events = []
    for comp_id, nodes in enumerate(components):
        nodes_list = list(nodes)
        comp_df = df.loc[nodes_list]
        events.append({
            "comp_id": comp_id,
            "start": comp_df["start_datetime"].min(),
            "end": comp_df["end_datetime"].max(),
            "nodes": nodes_list,
            "label": 1 if comp_df["manual_label"].sum() > 0 else 0,
        })
    return events


def merge_burst_intervals(bursts_df):
    """Merge overlapping burst intervals into contiguous events.

    This provides a **fair temporal baseline** by performing the same
    interval-union that the graph implicitly achieves via connected
    components, but using only 1-D temporal overlap (no graph needed).

    The merge criterion is strict overlap: ``start_j < end_i``.

    Parameters
    ----------
    bursts_df : pd.DataFrame
        Subset of the dataset containing only burst rows
        (``manual_label == 1``).  Must have ``start_datetime``,
        ``end_datetime``.

    Returns
    -------
    list[dict]
        Each dict has keys ``start``, ``end``.
    """
    if len(bursts_df) == 0:
        return []

    sorted_bursts = bursts_df.sort_values("start_datetime")
    events = []
    current_start = sorted_bursts.iloc[0]["start_datetime"]
    current_end = sorted_bursts.iloc[0]["end_datetime"]

    for i in range(1, len(sorted_bursts)):
        row = sorted_bursts.iloc[i]
        if row["start_datetime"] < current_end:          # strict overlap
            current_end = max(current_end, row["end_datetime"])
        else:
            events.append({"start": current_start, "end": current_end})
            current_start = row["start_datetime"]
            current_end = row["end_datetime"]

    events.append({"start": current_start, "end": current_end})
    return events


# ---------------------------------------------------------------------------
# Auxiliary Metrics
# ---------------------------------------------------------------------------

def compute_event_purity(start_local, end_local, start_global, end_global):
    """Fraction of the *global* event covered by the local observation."""
    intersection_start = max(start_local, start_global)
    intersection_end = min(end_local, end_global)
    if intersection_start >= intersection_end:
        return 0.0
    intersection_sec = (intersection_end - intersection_start).total_seconds()
    global_sec = (end_global - start_global).total_seconds()
    return intersection_sec / global_sec if global_sec > 0 else 0.0


def compute_event_expansion(start_local, end_local, start_global, end_global):
    """Ratio of global event duration to local event duration."""
    global_sec = (end_global - start_global).total_seconds()
    local_sec = (end_local - start_local).total_seconds()
    return global_sec / local_sec if local_sec > 0 else 0.0


# ---------------------------------------------------------------------------
# Split Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(df, train_idx, test_idx, window_minutes=15,
                   iot_threshold=0.5, is_temporal_baseline=False):
    """Evaluate one train/test split (graph or temporal baseline).

    Parameters
    ----------
    df : pd.DataFrame
    train_idx, test_idx : pd.Index
    window_minutes : int
    iot_threshold : float
        Minimum IoU / IoL to count as a match.
    is_temporal_baseline : bool
        If *True*, the "global events" are constructed by merging
        overlapping burst intervals in 1-D time (no graph).
        This is the **fair** temporal coincidence baseline.

    Returns
    -------
    dict or None
        Mean metrics across test antennas.
    """
    graph_stats = {}

    # --- Leakage guardrail ---
    assert len(train_idx.intersection(test_idx)) == 0, \
        "LEAKAGE: train_idx and test_idx overlap!"
    # Semantic check: target station must not appear in train data
    train_antennas = set(df.loc[train_idx, "antenna"].unique())
    test_antennas_set = set(df.loc[test_idx, "antenna"].unique())
    assert train_antennas.isdisjoint(test_antennas_set), \
        f"LEAKAGE: test antenna(s) {train_antennas & test_antennas_set} found in train!"

    if is_temporal_baseline:
        # --- Fair temporal baseline ---
        # Merge overlapping train bursts into contiguous events,
        # mirroring what the graph does via connected components.
        train_bursts = df.loc[train_idx][df.loc[train_idx, "manual_label"] == 1]
        E_g_bursts = merge_burst_intervals(train_bursts)
    else:
        # --- Graph baseline ---
        train_df = df.loc[train_idx]
        train_G, train_comps = build_graph(
            train_df, window_minutes
        )
        # Pass train-only df to prevent accidental test-label leakage
        E_g = extract_global_events(train_df, train_comps)
        E_g_bursts = [e for e in E_g if e["label"] == 1]

        comp_sizes = [len(c) for c in train_comps]
        graph_stats = {
            "edges": train_G.number_of_edges(),
            "components": len(train_comps),
            "mean_comp_size": (
                np.mean(comp_sizes) if comp_sizes else 0
            ),
            "lcf": (
                max(comp_sizes) / train_G.number_of_nodes()
                if comp_sizes and train_G.number_of_nodes() > 0
                else 0
            ),
        }

    test_antennas = df.loc[test_idx, "antenna"].unique()
    metrics = []

    for ant in test_antennas:
        ant_test_idx = df.loc[test_idx][
            df.loc[test_idx, "antenna"] == ant
        ].index
        E_h = extract_station_events(df, ant_test_idx, window_minutes)

        if len(E_h) == 0:
            continue

        recovered_events_iou = 0
        recovered_events_iol = 0
        fragmented_events = 0

        event_purities = []
        event_expansions = []

        for eh in E_h:
            overlaps_iou = []
            overlaps_iol = []
            best_iol = -1
            best_eg = None

            for eg in E_g_bursts:
                iou = compute_temporal_iou(
                    eh["start"], eh["end"], eg["start"], eg["end"]
                )
                iol = compute_event_coverage(
                    eh["start"], eh["end"], eg["start"], eg["end"]
                )
                if iou > iot_threshold:
                    overlaps_iou.append(eg)
                if iol > iot_threshold:
                    overlaps_iol.append(eg)
                if iol > best_iol:
                    best_iol = iol
                    best_eg = eg

            if len(overlaps_iou) > 0:
                recovered_events_iou += 1
            if len(overlaps_iol) > 0:
                recovered_events_iol += 1
                if len(overlaps_iol) > 1:
                    fragmented_events += 1

                pur = compute_event_purity(
                    eh["start"], eh["end"],
                    best_eg["start"], best_eg["end"],
                )
                exp = compute_event_expansion(
                    eh["start"], eh["end"],
                    best_eg["start"], best_eg["end"],
                )
                event_purities.append(pur)
                event_expansions.append(exp)

        err_iou = recovered_events_iou / len(E_h)
        err_iol = recovered_events_iol / len(E_h)
        fr = fragmented_events / len(E_h)
        avg_pur = np.mean(event_purities) if event_purities else 0.0
        avg_exp = np.mean(event_expansions) if event_expansions else 0.0

        station_metrics = {
            "station": ant,
            "nodes": len(ant_test_idx),
            "events": len(E_h),
            "ERR_IoU": err_iou,
            "ERR_IoL": err_iol,
            "Event_Purity": avg_pur,
            "Event_Expansion": avg_exp,
            "Fragmentation_Rate": fr,
        }
        station_metrics.update(graph_stats)
        metrics.append(station_metrics)

    if not metrics:
        return None

    return pd.DataFrame(metrics).mean(numeric_only=True).to_dict()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(data, num_bootstraps=10000):
    """Bootstrap 95 % CI for the mean."""
    if len(data) == 0:
        return 0, 0, 0
    values = np.array(data)
    bootstrapped = np.random.choice(
        values, size=(num_bootstraps, len(values)), replace=True
    )
    means = np.mean(bootstrapped, axis=1)
    return (
        np.mean(values),
        np.percentile(means, 2.5),
        np.percentile(means, 97.5),
    )


# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------

def run_evaluation(df, window_minutes, thresholds):
    all_results = []

    for thresh in thresholds:
        logging.info(f"\n=============================================")
        logging.info(f"Running evaluation with Threshold = {thresh}")
        logging.info(f"=============================================")

        # 1. Random Station Split (Graph vs Temporal) ----------------------
        logging.info(
            "Running Baseline 2: Random Station Split (Graph vs Temporal)"
        )
        antennas = df["antenna"].unique()
        station_errs_iol_graph, station_errs_iol_temp = [], []

        np.random.seed(42)
        for _ in range(5):
            test_ants = np.random.choice(
                antennas,
                size=max(1, int(len(antennas) * 0.2)),
                replace=False,
            )
            test_idx = df[df["antenna"].isin(test_ants)].index
            train_idx = df.index.difference(test_idx)

            # Graph
            metrics_g = evaluate_split(
                df, train_idx, test_idx, window_minutes, thresh,
                is_temporal_baseline=False,
            )
            if metrics_g:
                station_errs_iol_graph.append(metrics_g["ERR_IoL"])

            # Temporal
            metrics_t = evaluate_split(
                df, train_idx, test_idx, window_minutes, thresh,
                is_temporal_baseline=True,
            )
            if metrics_t:
                station_errs_iol_temp.append(metrics_t["ERR_IoL"])

        all_results.append({
            "Threshold": thresh,
            "split": "random_station_graph",
            "station": "mean_20pct_stations",
            "ERR_IoL": (
                np.mean(station_errs_iol_graph)
                if station_errs_iol_graph else 0
            ),
        })
        all_results.append({
            "Threshold": thresh,
            "split": "random_station_temporal",
            "station": "mean_20pct_stations",
            "ERR_IoL": (
                np.mean(station_errs_iol_temp)
                if station_errs_iol_temp else 0
            ),
        })

        # 2. EG-LOSO (Graph vs Temporal paired) ---------------------------
        logging.info("Running EG-LOSO (Complete Instrument Loss)")
        top_stations = (
            df[df["manual_label"] == 1]["antenna"]
            .value_counts()
            .head(15)
            .index
        )

        eg_iol_graph = []
        eg_iol_temp = []
        edges_list, comps_list, mean_comp_sizes, lcfs = [], [], [], []

        for ant in tqdm(top_stations, desc="EG-LOSO Stations"):
            test_idx = df[df["antenna"] == ant].index
            train_idx = df.index.difference(test_idx)

            # Semantic leakage guardrail
            train_antennas = set(df.loc[train_idx, "antenna"].unique())
            assert ant not in train_antennas, \
                f"LEAKAGE: target station {ant} found in train data!"

            # Graph
            metrics_g = evaluate_split(
                df, train_idx, test_idx, window_minutes, thresh,
                is_temporal_baseline=False,
            )
            if metrics_g:
                eg_iol_graph.append(metrics_g["ERR_IoL"])
                edges_list.append(metrics_g.get("edges", 0))
                comps_list.append(metrics_g.get("components", 0))
                mean_comp_sizes.append(
                    metrics_g.get("mean_comp_size", 0)
                )
                lcfs.append(metrics_g.get("lcf", 0))
                all_results.append({
                    "Threshold": thresh,
                    "split": "EG_LOSO_graph",
                    "station": ant,
                    "ERR_IoL": metrics_g["ERR_IoL"],
                    "Purity": metrics_g["Event_Purity"],
                    "Expansion": metrics_g["Event_Expansion"],
                    "split_integrity": {
                        "train_stations": len(train_antennas),
                        "test_station": ant,
                        "overlap": 0,
                    },
                })
            else:
                eg_iol_graph.append(0)

            # Temporal
            metrics_t = evaluate_split(
                df, train_idx, test_idx, window_minutes, thresh,
                is_temporal_baseline=True,
            )
            if metrics_t:
                eg_iol_temp.append(metrics_t["ERR_IoL"])
                all_results.append({
                    "Threshold": thresh,
                    "split": "EG_LOSO_temporal",
                    "station": ant,
                    "ERR_IoL": metrics_t["ERR_IoL"],
                    "Purity": metrics_t["Event_Purity"],
                    "Expansion": metrics_t["Event_Expansion"],
                })
            else:
                eg_iol_temp.append(0)

        # Statistical Test: Delta IoL
        deltas = np.array(eg_iol_graph) - np.array(eg_iol_temp)
        d_mean, d_low, d_high = bootstrap_ci(deltas, num_bootstraps=10000)

        logging.info(
            f"Threshold {thresh} -> Delta IoL (Graph - Temporal) = "
            f"{d_mean:.4f} 95% CI: [{d_low:.4f}, {d_high:.4f}]"
        )

        all_results.append({
            "Threshold": thresh,
            "split": "EG_LOSO_SUMMARY",
            "station": "mean_all",
            "ERR_IoL_Graph": np.mean(eg_iol_graph),
            "ERR_IoL_Temporal": np.mean(eg_iol_temp),
            "Delta_IoL_Mean": d_mean,
            "Delta_IoL_CI_low": d_low,
            "Delta_IoL_CI_high": d_high,
            "Edges": np.mean(edges_list),
            "Components": np.mean(comps_list),
            "Mean_Comp_Size": np.mean(mean_comp_sizes),
            "LCF": np.mean(lcfs),
        })

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "all"], default="test")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config.get("benchmark", {}).get(
        "dataset", "i4ds/ecallisto_radio_sunburst"
    )
    window_minutes = config.get("benchmark", {}).get("window_minutes", 15)

    logging.info("Loading dataset...")
    ds = load_dataset(dataset_name)
    cols = ["manual_label", "start_datetime", "antenna"]
    df = ds[args.split].to_pandas()[cols]

    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    if "end_datetime" not in df.columns:
        df["end_datetime"] = df["start_datetime"] + pd.Timedelta(
            minutes=window_minutes
        )

    df = df.sort_values("start_datetime").reset_index(drop=True)

    thresholds = [0.3, 0.5, 0.7, 0.9]
    all_results = run_evaluation(df, window_minutes, thresholds)

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(REPORTS_DIR / "eg_loso_final_results.csv", index=False)

    logging.info(
        "\nDone! Results saved to event_graph/reports/eg_loso_final_results.csv"
    )


if __name__ == "__main__":
    main()
