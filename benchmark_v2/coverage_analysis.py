import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from datasets import load_dataset
from pathlib import Path
import logging

from graph_utils import build_graph

logging.basicConfig(level=logging.INFO, format="%(message)s")

def bootstrap_spearman(x, y, n_bootstraps=1000):
    corrs = []
    indices = np.arange(len(x))
    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(indices, size=len(indices), replace=True)
        # Add small noise to avoid identical values breaking spearmanr when sample is too uniform
        boot_x = x[boot_idx] + np.random.normal(0, 1e-10, len(boot_idx))
        boot_y = y[boot_idx] + np.random.normal(0, 1e-10, len(boot_idx))
        corr, _ = spearmanr(boot_x, boot_y)
        if not np.isnan(corr):
            corrs.append(corr)
    if not corrs: return 0, 0, 0
    return np.mean(corrs), np.percentile(corrs, 2.5), np.percentile(corrs, 97.5)

def get_ucr(df, window_minutes):
    G, components = build_graph(df, window_minutes)
    if not components: return 0.0
    
    false_clusters = 0
    for comp in components:
        nodes_list = list(comp)
        if df.loc[nodes_list, "manual_label"].sum() == 0:
            false_clusters += 1
            
    return false_clusters / len(components)

def main():
    logging.info("Loading EG-LOSO results...")
    loso_path = Path("event_graph/reports/eg_loso_results.csv")
    if not loso_path.exists():
        logging.error("ERROR: File event_graph/reports/eg_loso_results.csv not found.")
        logging.error("Please run eg_loso_core.py first to generate the required reports.")
        return
    loso_df = pd.read_csv(loso_path)
    # Filter only individual stations (exclude Random splits and Summary)
    st_df = loso_df[loso_df["split"] == "EG_LOSO"].copy()
    
    logging.info("Loading dataset to compute coverage...")
    ds = load_dataset("i4ds/ecallisto_radio_sunburst", split="test")
    df = ds.to_pandas()[["manual_label", "start_datetime", "antenna"]]
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    df["end_datetime"] = df["start_datetime"] + pd.Timedelta(minutes=15)
    df = df.sort_values("start_datetime").reset_index(drop=True)
    
    # 1. Topological dependency (Graph degree)
    logging.info("Building base graph for topological dependency...")
    G, _ = build_graph(df, 15)
    degrees = dict(G.degree())
    df["degree"] = df.index.map(degrees)
    
    # Pre-calculate temporal overlaps (independent of graph edges, just physical presence of ANY other station)
    # Actually, if other stations overlap in [start, start+15m], it's practically the same condition as the graph,
    # but the physical overlap counts HOW MANY unique stations were observing simultaneously.
    logging.info("Calculating physical coverage (temporal overlap & co-observers)...")
    
    station_metrics = []
    
    # For performance, we can just use the graph edges since our graph is an exact representation 
    # of "which other stations overlapped temporally". 
    # Graph degree = total overlapping nodes from other stations = observation density.
    # Unique neighbors' antennas = unique co-observing stations.
    for st in st_df["station"]:
        st_nodes = df[df["antenna"] == st].index.tolist()
        
        # Confound control
        station_size = len(st_nodes)
        event_count = df.loc[st_nodes, "manual_label"].sum()
        
        # Topological
        mean_degree = np.mean([degrees[n] for n in st_nodes]) if st_nodes else 0
        
        # Physical Coverage (Temporal Overlap & Co-observers)
        co_observers = []
        for n in st_nodes:
            neighbors = list(G.neighbors(n))
            neighbor_stations = df.loc[neighbors, "antenna"].unique()
            co_observers.extend(neighbor_stations)
            
        unique_co_observers = len(set(co_observers))
        
        # Average number of unique stations co-observing per node
        avg_simultaneous_stations = np.mean([len(df.loc[list(G.neighbors(n)), "antenna"].unique()) for n in st_nodes]) if st_nodes else 0
        
        err = float(st_df[st_df["station"] == st]["ERR"].values[0])
        
        station_metrics.append({
            "station": st,
            "ERR": err,
            "station_size": station_size,
            "event_count": event_count,
            "graph_degree": mean_degree,
            "unique_co_observers": unique_co_observers,
            "temporal_overlap": avg_simultaneous_stations
        })
        
    metrics_df = pd.DataFrame(station_metrics)
    metrics_df.to_csv("event_graph/reports/coverage_metrics.csv", index=False)
    
    logging.info("\n--- 3. ERR Explanation (Spearman Correlation) ---")
    y = metrics_df["ERR"].values
    
    comparisons = [
        ("Temporal Overlap", metrics_df["temporal_overlap"].values),
        ("Unique Co-observers", metrics_df["unique_co_observers"].values),
        ("Graph Degree", metrics_df["graph_degree"].values),
        ("Station Size (Confound)", metrics_df["station_size"].values),
        ("Event Count (Confound)", metrics_df["event_count"].values)
    ]
    
    for name, x in comparisons:
        rho, p = spearmanr(x, y)
        mean_rho, ci_l, ci_h = bootstrap_spearman(x, y)
        logging.info(f"{name}:")
        logging.info(f"  Spearman rho = {rho:.3f}")
        logging.info(f"  95% CI       = [{ci_l:.3f}, {ci_h:.3f}]")
        logging.info(f"  p-value      = {p:.4f}\n")
        
    logging.info("--- 4. Stratified EG-LOSO ---")
    # Stratify by Temporal Overlap
    q33 = metrics_df["temporal_overlap"].quantile(0.33)
    q66 = metrics_df["temporal_overlap"].quantile(0.66)
    
    low_cov = metrics_df[metrics_df["temporal_overlap"] <= q33]["ERR"].mean()
    med_cov = metrics_df[(metrics_df["temporal_overlap"] > q33) & (metrics_df["temporal_overlap"] <= q66)]["ERR"].mean()
    high_cov = metrics_df[metrics_df["temporal_overlap"] > q66]["ERR"].mean()
    
    logging.info(f"Low Coverage Stations ERR:    {low_cov:.3f}")
    logging.info(f"Medium Coverage Stations ERR: {med_cov:.3f}")
    logging.info(f"High Coverage Stations ERR:   {high_cov:.3f}\n")
    
    logging.info("--- 6. Noise Robustness (Unverified Cluster Rate) ---")
    ucr_15 = get_ucr(df, 15)
    ucr_60 = get_ucr(df, 60)
    
    logging.info(f"UCR @ 15 min: {ucr_15:.2%}")
    logging.info(f"UCR @ 60 min: {ucr_60:.2%}")
    
    if ucr_60 > ucr_15 * 2:
        logging.info("-> WARNING: Graph aggressively merges noise at higher windows.")
    else:
        logging.info("-> SUCCESS: Noise component rate is stable across windows.")

if __name__ == "__main__":
    main()
