"""
Unified graph construction and temporal metric utilities for FlareSense-v2.

This module is the **single source of truth** for graph construction and
core temporal metrics. ALL benchmark modules MUST import from here to
ensure mathematical consistency across the audit pipeline.

Architecture:
    Raw spectrograms
        → build_graph()        → Topology / Null / Adversarial tests
        → build_graph_events() → EG-LOSO / Structural / Stress tests

    compute_temporal_iou()    — IoU  (Intersection / Union)
    compute_event_coverage()  — IoL  (Intersection / Local)
    evaluate_topology()       — GCF, component count, largest component
"""

import pandas as pd
import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph(df, window_minutes=15, cross_station_only=True, weighted=False):
    """Build a temporal overlap graph from spectrogram observations.

    Connects samples whose time windows [start, end) overlap.
    Uses the DataFrame's index as node identifiers.

    **Important**: ``df`` MUST be sorted by ``start_datetime`` before calling.
    The early-termination optimisation (``break``) relies on this ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``start_datetime``, ``antenna``.
        Optionally ``end_datetime``; if absent, computed as
        ``start_datetime + window_minutes``.
    window_minutes : int
        Default window length in minutes (used only when ``end_datetime``
        is absent).
    cross_station_only : bool
        If *True* (default), edges connect only samples from **different**
        stations.  If *False*, all temporally overlapping samples are
        connected regardless of station.
    weighted : bool
        If *True*, edges carry ``jaccard`` (temporal Jaccard similarity)
        and ``overlap`` (overlap duration in seconds) attributes.

    Returns
    -------
    G : nx.Graph
    components : list[set]
        Connected components of *G*.
    """
    G = nx.Graph()
    indices = df.index.values
    G.add_nodes_from(indices)

    start_times = df["start_datetime"].values
    if "end_datetime" in df.columns:
        end_times = df["end_datetime"].values
    else:
        end_times = (
            df["start_datetime"] + pd.Timedelta(minutes=window_minutes)
        ).values

    antennas = df["antenna"].values
    edges_to_add = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            # Early termination: sorted order guarantees no further overlaps
            if start_times[j] >= end_times[i]:
                break

            if cross_station_only and antennas[i] == antennas[j]:
                continue

            if weighted:
                s1, e1 = start_times[i], end_times[i]
                s2, e2 = start_times[j], end_times[j]

                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                overlap_dur = (
                    (overlap_end - overlap_start) / np.timedelta64(1, "s")
                )

                union_start = min(s1, s2)
                union_end = max(e1, e2)
                union_dur = (
                    (union_end - union_start) / np.timedelta64(1, "s")
                )

                jaccard = overlap_dur / union_dur if union_dur > 0 else 0
                edges_to_add.append(
                    (indices[i], indices[j],
                     {"jaccard": jaccard, "overlap": overlap_dur})
                )
            else:
                edges_to_add.append((indices[i], indices[j]))

    G.add_edges_from(edges_to_add)
    components = list(nx.connected_components(G))
    return G, components


def build_graph_events(df, window_minutes=15, cross_station_only=True):
    """Build a graph and return per-row component labels.

    Convenience wrapper around :func:`build_graph` that additionally maps
    each DataFrame row to its connected-component ID.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by ``start_datetime``.
    window_minutes : int
    cross_station_only : bool

    Returns
    -------
    labels : pd.Series
        Component ID for each row in *df*.
    G : nx.Graph
    components : list[set]
    """
    G, components = build_graph(df, window_minutes, cross_station_only)

    comp_map = {}
    for comp_id, nodes in enumerate(components):
        for node in nodes:
            comp_map[node] = comp_id

    labels = df.index.map(comp_map)
    return labels, G, components


# ---------------------------------------------------------------------------
# Temporal Metrics
# ---------------------------------------------------------------------------

def compute_temporal_iou(start1, end1, start2, end2):
    """Temporal Intersection-over-Union of two intervals.

    Parameters
    ----------
    start1, end1 : datetime-like
        First interval ``[start1, end1)``.
    start2, end2 : datetime-like
        Second interval ``[start2, end2)``.

    Returns
    -------
    float
        IoU ∈ [0, 1].
    """
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    if intersection_start >= intersection_end:
        return 0.0
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    intersection_sec = (intersection_end - intersection_start).total_seconds()
    union_sec = (union_end - union_start).total_seconds()
    return intersection_sec / union_sec if union_sec > 0 else 0.0


def compute_event_coverage(start_local, end_local, start_global, end_global):
    """IoL (Intersection-over-Local) = Intersection / Local duration.

    Answers: *"What fraction of the local observation was covered by
    the global event?"*

    Parameters
    ----------
    start_local, end_local : datetime-like
        Local station event interval.
    start_global, end_global : datetime-like
        Global graph event interval.

    Returns
    -------
    float
        IoL ∈ [0, 1].
    """
    intersection_start = max(start_local, start_global)
    intersection_end = min(end_local, end_global)
    if intersection_start >= intersection_end:
        return 0.0
    intersection_sec = (intersection_end - intersection_start).total_seconds()
    local_sec = (end_local - start_local).total_seconds()
    return intersection_sec / local_sec if local_sec > 0 else 0.0


# ---------------------------------------------------------------------------
# Topology Helpers
# ---------------------------------------------------------------------------

def evaluate_topology(G, components, total_nodes):
    """Compute basic topology metrics for a graph.

    Parameters
    ----------
    G : nx.Graph
    components : list[set]
    total_nodes : int

    Returns
    -------
    dict
        Keys: ``num_components``, ``largest_component``, ``gcf``.
    """
    if not components:
        return {
            "num_components": 0,
            "largest_component": 0,
            "gcf": 0.0,
        }
    largest = max(len(c) for c in components)
    return {
        "num_components": len(components),
        "largest_component": largest,
        "gcf": largest / total_nodes if total_nodes > 0 else 0.0,
    }
