"""
Unified graph construction utilities for the FlareSense-v2 benchmark.

This module provides a single, canonical graph construction function
to avoid the duplication and inconsistency issues across modules.
All benchmark modules should import from here instead of defining
their own build_graph functions.
"""

import pandas as pd
import numpy as np
import networkx as nx


def build_graph(df, window_minutes=15, cross_station_only=True, weighted=False):
    """Build a temporal overlap graph from spectrogram observations.

    Connects samples whose time windows overlap. The graph uses the DataFrame's
    index as node identifiers.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'start_datetime', 'antenna'.
        Optionally 'end_datetime' (if absent, computed as start + window_minutes).
    window_minutes : int
        Default window length in minutes (used only if 'end_datetime' is absent).
    cross_station_only : bool
        If True (default), only connect samples from *different* stations.
        If False, connect all temporally overlapping samples regardless of station.
    weighted : bool
        If True, edges carry 'jaccard' and 'overlap' attributes (temporal Jaccard
        similarity and overlap duration in seconds).

    Returns
    -------
    G : nx.Graph
        The constructed graph.
    components : list of sets
        Connected components of G.
    """
    G = nx.Graph()
    indices = df.index.values
    G.add_nodes_from(indices)

    start_times = df["start_datetime"].values
    if "end_datetime" in df.columns:
        end_times = df["end_datetime"].values
    else:
        end_times = (df["start_datetime"] + pd.Timedelta(minutes=window_minutes)).values

    antennas = df["antenna"].values
    edges_to_add = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if start_times[j] >= end_times[i]:
                break

            if cross_station_only and antennas[i] == antennas[j]:
                continue

            if weighted:
                s1, e1 = start_times[i], end_times[i]
                s2, e2 = start_times[j], end_times[j]

                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                overlap_dur = (overlap_end - overlap_start) / np.timedelta64(1, 's')

                union_start = min(s1, s2)
                union_end = max(e1, e2)
                union_dur = (union_end - union_start) / np.timedelta64(1, 's')

                jaccard = overlap_dur / union_dur if union_dur > 0 else 0
                edges_to_add.append((indices[i], indices[j], {"jaccard": jaccard, "overlap": overlap_dur}))
            else:
                edges_to_add.append((indices[i], indices[j]))

    if weighted:
        G.add_edges_from(edges_to_add)
    else:
        G.add_edges_from(edges_to_add)

    components = list(nx.connected_components(G))
    return G, components
