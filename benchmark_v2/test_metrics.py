"""
Unit tests for benchmark_v2 metric functions.

Tests cover:
  1. Temporal metrics (IoU, IoL, Purity, Expansion)
  2. Over-merging prevention in extract_station_events
  3. Fair temporal baseline merging
  4. Graph construction determinism
"""

import pandas as pd
import numpy as np
from graph_utils import compute_temporal_iou, compute_event_coverage, build_graph
from eg_loso_core import (
    compute_event_purity,
    compute_event_expansion,
    extract_station_events,
    merge_burst_intervals,
)


def run_test_case(name, sl, el, sg, eg, exp_iol, exp_iou, exp_pur, exp_exp):
    iol = compute_event_coverage(sl, el, sg, eg)
    iou = compute_temporal_iou(sl, el, sg, eg)
    pur = compute_event_purity(sl, el, sg, eg)
    exp = compute_event_expansion(sl, el, sg, eg)

    ok = True
    for label, got, expected in [
        ("IoL", iol, exp_iol),
        ("IoU", iou, exp_iou),
        ("Purity", pur, exp_pur),
        ("Expansion", exp, exp_exp),
    ]:
        if abs(got - expected) > 0.01:
            print(f"  FAIL {label}: got {got:.3f}, expected {expected}")
            ok = False
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    return ok


def test_temporal_metrics():
    print("\n=== Temporal Metrics ===")
    t1 = pd.to_datetime("2021-01-01 10:00:00")
    t2 = pd.to_datetime("2021-01-01 10:15:00")

    all_ok = True

    # Case 1: Exact match
    all_ok &= run_test_case("Exact Match", t1, t2, t1, t2, 1.0, 1.0, 1.0, 1.0)

    # Case 2: No overlap
    t3 = pd.to_datetime("2021-01-01 12:00:00")
    t4 = pd.to_datetime("2021-01-01 12:15:00")
    all_ok &= run_test_case("No Overlap", t1, t2, t3, t4, 0.0, 0.0, 0.0, 1.0)

    # Case 3: Global smaller than local
    sg3 = pd.to_datetime("2021-01-01 10:05:00")
    eg3 = pd.to_datetime("2021-01-01 10:10:00")
    all_ok &= run_test_case("Global < Local", t1, t2, sg3, eg3, 0.333, 0.333, 1.0, 0.333)

    # Case 4: Local inside massive global
    sg4 = pd.to_datetime("2021-01-01 09:30:00")
    eg4 = pd.to_datetime("2021-01-01 11:30:00")
    all_ok &= run_test_case("Local inside Global", t1, t2, sg4, eg4, 1.0, 0.125, 0.125, 8.0)

    # Case 5: Large global covering multiple locals
    sg5 = pd.to_datetime("2021-01-01 10:00:00")
    eg5 = pd.to_datetime("2021-01-01 11:15:00")
    all_ok &= run_test_case("Global covers multiple", t1, t2, sg5, eg5, 1.0, 0.2, 0.2, 5.0)

    return all_ok


def test_no_over_merging():
    """Verify that extract_station_events does NOT over-merge
    events separated by a gap."""
    print("\n=== Over-Merging Prevention ===")
    all_ok = True

    # Two bursts separated by 5 min gap (should NOT merge)
    # Burst 1: 10:00-10:15, Burst 2: 10:20-10:35
    df = pd.DataFrame({
        "start_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-01-01 10:20:00",
        ]),
        "end_datetime": pd.to_datetime([
            "2021-01-01 10:15:00",
            "2021-01-01 10:35:00",
        ]),
        "manual_label": [1, 1],
        "antenna": ["ST_A", "ST_A"],
    })

    events = extract_station_events(df, df.index, window_minutes=15)
    ok = len(events) == 2
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Two non-overlapping bursts -> {len(events)} events (expected 2)")
    all_ok &= ok

    # Two bursts that DO overlap (should merge)
    df2 = pd.DataFrame({
        "start_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-01-01 10:10:00",
        ]),
        "end_datetime": pd.to_datetime([
            "2021-01-01 10:15:00",
            "2021-01-01 10:25:00",
        ]),
        "manual_label": [1, 1],
        "antenna": ["ST_A", "ST_A"],
    })

    events2 = extract_station_events(df2, df2.index, window_minutes=15)
    ok2 = len(events2) == 1
    status2 = "PASS" if ok2 else "FAIL"
    print(f"[{status2}] Two overlapping bursts -> {len(events2)} events (expected 1)")
    all_ok &= ok2

    # Edge case: exactly touching (start_2 == end_1) -> should NOT merge
    # (graph uses strict < not <=)
    df3 = pd.DataFrame({
        "start_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-01-01 10:15:00",
        ]),
        "end_datetime": pd.to_datetime([
            "2021-01-01 10:15:00",
            "2021-01-01 10:30:00",
        ]),
        "manual_label": [1, 1],
        "antenna": ["ST_A", "ST_A"],
    })

    events3 = extract_station_events(df3, df3.index, window_minutes=15)
    ok3 = len(events3) == 2
    status3 = "PASS" if ok3 else "FAIL"
    print(f"[{status3}] Exactly touching bursts -> {len(events3)} events (expected 2)")
    all_ok &= ok3

    return all_ok


def test_merge_burst_intervals():
    """Test the fair temporal baseline merge."""
    print("\n=== Temporal Baseline Merge ===")
    all_ok = True

    # 3 overlapping + 1 separate -> 2 merged events
    df = pd.DataFrame({
        "start_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-01-01 10:05:00",
            "2021-01-01 10:10:00",
            "2021-01-01 11:00:00",
        ]),
        "end_datetime": pd.to_datetime([
            "2021-01-01 10:15:00",
            "2021-01-01 10:20:00",
            "2021-01-01 10:25:00",
            "2021-01-01 11:15:00",
        ]),
        "manual_label": [1, 1, 1, 1],
    })

    merged = merge_burst_intervals(df)
    ok = len(merged) == 2
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] 3 overlapping + 1 separate -> {len(merged)} events (expected 2)")
    all_ok &= ok

    if ok:
        ok2 = merged[0]["end"] == pd.to_datetime("2021-01-01 10:25:00")
        status2 = "PASS" if ok2 else "FAIL"
        print(f"[{status2}] First event end = {merged[0]['end']} (expected 10:25)")
        all_ok &= ok2

    # Empty input
    empty_df = pd.DataFrame({
        "start_datetime": pd.Series(dtype="datetime64[ns]"),
        "end_datetime": pd.Series(dtype="datetime64[ns]"),
        "manual_label": pd.Series(dtype=int),
    })
    merged_empty = merge_burst_intervals(empty_df)
    ok3 = len(merged_empty) == 0
    status3 = "PASS" if ok3 else "FAIL"
    print(f"[{status3}] Empty input -> {len(merged_empty)} events (expected 0)")
    all_ok &= ok3

    return all_ok


def test_graph_determinism():
    """Two identical build_graph calls must produce the same result."""
    print("\n=== Graph Determinism ===")

    df = pd.DataFrame({
        "start_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-01-01 10:05:00",
            "2021-01-01 10:10:00",
            "2021-01-01 11:00:00",
        ]),
        "end_datetime": pd.to_datetime([
            "2021-01-01 10:15:00",
            "2021-01-01 10:20:00",
            "2021-01-01 10:25:00",
            "2021-01-01 11:15:00",
        ]),
        "antenna": ["A", "B", "A", "B"],
        "manual_label": [1, 1, 1, 1],
    }).sort_values("start_datetime").reset_index(drop=True)

    G1, c1 = build_graph(df, 15)
    G2, c2 = build_graph(df, 15)

    same_nodes = set(G1.nodes()) == set(G2.nodes())
    same_edges = set(G1.edges()) == set(G2.edges())
    same_comps = len(c1) == len(c2)

    ok = same_nodes and same_edges and same_comps
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Deterministic: nodes={same_nodes}, edges={same_edges}, comps={same_comps}")
    return ok


def test_metric_invariance():
    """Shuffling station order should not affect metrics."""
    print("\n=== Metric Invariance (Station Order) ===")

    base = pd.DataFrame({
        "start_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-01-01 10:05:00",
            "2021-01-01 10:02:00",
            "2021-01-01 11:00:00",
        ]),
        "end_datetime": pd.to_datetime([
            "2021-01-01 10:15:00",
            "2021-01-01 10:20:00",
            "2021-01-01 10:17:00",
            "2021-01-01 11:15:00",
        ]),
        "antenna": ["A", "B", "C", "A"],
        "manual_label": [1, 1, 1, 0],
    })

    # Sort properly (required for build_graph)
    df1 = base.sort_values("start_datetime").reset_index(drop=True)
    G1, c1 = build_graph(df1, 15)

    # Shuffle rows then re-sort
    df2 = base.sample(frac=1, random_state=99).sort_values("start_datetime").reset_index(drop=True)
    G2, c2 = build_graph(df2, 15)

    ok = G1.number_of_edges() == G2.number_of_edges() and len(c1) == len(c2)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Same edges ({G1.number_of_edges()} vs {G2.number_of_edges()}) "
          f"and components ({len(c1)} vs {len(c2)}) after shuffle")
    return ok


if __name__ == "__main__":
    results = []
    results.append(("Temporal Metrics", test_temporal_metrics()))
    results.append(("Over-Merging Prevention", test_no_over_merging()))
    results.append(("Temporal Baseline Merge", test_merge_burst_intervals()))
    results.append(("Graph Determinism", test_graph_determinism()))
    results.append(("Metric Invariance", test_metric_invariance()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_passed = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_passed &= ok

    if all_passed:
        print("\nAll tests PASSED.")
    else:
        print("\nSome tests FAILED!")
        exit(1)
