import pandas as pd
import numpy as np
import networkx as nx
from datasets import load_dataset
from pathlib import Path
import logging
from tqdm import tqdm
from graph_utils import build_graph, compute_temporal_iou
from eg_loso_core import extract_station_events, extract_global_events

logging.basicConfig(level=logging.INFO, format="%(message)s")
REPORTS_DIR = Path("event_graph/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def build_weighted_graph(df, window_minutes=15):
    """Build a weighted temporal overlap graph. Delegates to graph_utils."""
    G, _ = build_graph(df, window_minutes, cross_station_only=True, weighted=True)
    return G

def evaluate_components_tracking(df, train_idx, test_idx, train_comps, window_minutes=15, iot_threshold=0.5):
    # This evaluates one EG-LOSO fold and tracks which global events were matched
    assert len(train_idx.intersection(test_idx)) == 0, "LEAKAGE: train/test overlap!"
    # Use train-only df to prevent accidental test-label leakage
    train_df = df.loc[train_idx]
    E_g = extract_global_events(train_df, train_comps)
    E_g_bursts = [e for e in E_g if e["label"] == 1]
    
    test_antennas = df.loc[test_idx, "antenna"].unique()
    matched_eg_indices = set()
    metrics = []
    
    for ant in test_antennas:
        ant_test_idx = df.loc[test_idx][df.loc[test_idx, "antenna"] == ant].index
        E_h = extract_station_events(df, ant_test_idx, window_minutes)
        if len(E_h) == 0: continue
            
        recovered_events = 0
        for eh in E_h:
            overlaps = []
            for i, eg in enumerate(E_g_bursts):
                iot = compute_temporal_iou(eh["start"], eh["end"], eg["start"], eg["end"])
                if iot > iot_threshold:
                    overlaps.append(i)
            if len(overlaps) > 0:
                recovered_events += 1
                for idx in overlaps:
                    matched_eg_indices.add(idx)
                            
        err = recovered_events / len(E_h)
        metrics.append(err)
        
    mean_err = np.mean(metrics) if metrics else 0
    return mean_err, matched_eg_indices, E_g_bursts

def run_targeted_removal_eval(df, G, top_stations, window_minutes=15):
    errs = []
    for ant in top_stations:
        test_idx = df[df["antenna"] == ant].index
        train_idx = df.index.difference(test_idx)
        # Data lineage assertion: target station must be absent from training set
        assert ant not in set(df.loc[train_idx, "antenna"].unique()), f"LEAKAGE DETECTED: target station {ant} present in train_idx!"
        # SAFETY NOTE: G.subgraph(train_idx) is mathematically equivalent to
        # rebuilding from df.loc[train_idx] because edge weights depend only on
        # pairwise start/end times. No third-node information affects any edge.
        train_G = G.subgraph(train_idx)
        train_comps = list(nx.connected_components(train_G))
        mean_err, _, _ = evaluate_components_tracking(df, train_idx, test_idx, train_comps, window_minutes)
        errs.append(mean_err)
    return np.mean(errs)

def get_pruned_graph(G, strategy, p, betweenness=None):
    if p == 0: return G
    
    edges = list(G.edges(data=True))
    n_remove = int(len(edges) * p)
    
    if strategy == "random":
        np.random.shuffle(edges)
        edges_to_remove = edges[:n_remove]
    elif strategy == "weak":
        edges.sort(key=lambda x: x[2]["jaccard"]) # Ascending, lowest Jaccard first
        edges_to_remove = edges[:n_remove]
    elif strategy == "strong":
        edges.sort(key=lambda x: x[2]["jaccard"], reverse=True) # Descending, highest Jaccard first
        edges_to_remove = edges[:n_remove]
    elif strategy == "betweenness":
        # Requires betweenness dict
        edges_with_bc = [(u, v, betweenness.get((u, v), betweenness.get((v, u), 0))) for u, v, _ in edges]
        edges_with_bc.sort(key=lambda x: x[2], reverse=True) # Highest betweenness first
        edges_to_remove = edges_with_bc[:n_remove]
    else:
        raise ValueError("Invalid strategy")
        
    G_pruned = G.copy()
    G_pruned.remove_edges_from([(u, v) for u, v, _ in edges_to_remove])
    return G_pruned

def main():
    logging.info("Loading dataset...")
    ds = load_dataset("i4ds/ecallisto_radio_sunburst", split="test")
    cols = ["manual_label", "start_datetime", "antenna"]
    df = ds.to_pandas()[cols]
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    df["end_datetime"] = df["start_datetime"] + pd.Timedelta(minutes=15)
    df = df.sort_values("start_datetime").reset_index(drop=True)
    
    top_stations = df[df["manual_label"]==1]["antenna"].value_counts().head(15).index
    
    logging.info("Building weighted graph...")
    G = build_weighted_graph(df, 15)
    
    logging.info("Calculating approximate edge betweenness (k=1000)...")
    # k=1000 provides a good approximation for bridge identification quickly
    betweenness = nx.edge_betweenness_centrality(G, k=1000, seed=42)
    
    # ---------------------------------------------------------
    # Block A: Targeted Edge Removal
    # ---------------------------------------------------------
    logging.info("\n=== BLOCK A: Targeted Edge Removal ===")
    removal_results = []
    drop_rates = [0.10, 0.20, 0.30]
    
    # Baseline 0%
    base_err = run_targeted_removal_eval(df, G, top_stations, 15)
    removal_results.append({"Strategy": "Baseline", "DropRate": 0.0, "ERR_mean": base_err, "ERR_std": 0.0})
    
    strategies = ["weak", "strong", "betweenness", "random"]
    for strat in strategies:
        logging.info(f"Strategy: {strat}")
        for p in drop_rates:
            if strat == "random":
                errs = []
                for seed in range(5):
                    np.random.seed(42 + seed)
                    G_p = get_pruned_graph(G, strat, p, betweenness)
                    errs.append(run_targeted_removal_eval(df, G_p, top_stations, 15))
                removal_results.append({
                    "Strategy": strat, "DropRate": p, 
                    "ERR_mean": np.mean(errs), "ERR_std": np.std(errs)
                })
            else:
                G_p = get_pruned_graph(G, strat, p, betweenness)
                err = run_targeted_removal_eval(df, G_p, top_stations, 15)
                removal_results.append({
                    "Strategy": strat, "DropRate": p, 
                    "ERR_mean": err, "ERR_std": 0.0
                })
                
    pd.DataFrame(removal_results).to_csv(REPORTS_DIR / "struct_edge_removal.csv", index=False)
    
    # ---------------------------------------------------------
    # Block B: Event-Level Physics
    # ---------------------------------------------------------
    logging.info("\n=== BLOCK B: Event-Level Physics ===")
    # 1. Identify all True Global Events (using the full graph without masking)
    full_comps = list(nx.connected_components(G))
    E_g_all = extract_global_events(df, full_comps)
    E_g_true = [e for e in E_g_all if e["label"] == 1]
    
    # We will track hits per event across the 15 EG-LOSO folds
    event_hits = {i: 0 for i in range(len(E_g_true))}
    event_tests = {i: 0 for i in range(len(E_g_true))} # How many times was this event subject to masking?
    
    # We need to map E_g_true to the train_comps in each fold.
    # To do this robustly, we track by node overlap.
    for ant in tqdm(top_stations, desc="EG-LOSO Folds"):
        test_idx = df[df["antenna"] == ant].index
        train_idx = df.index.difference(test_idx)
        train_G = G.subgraph(train_idx)
        train_comps = list(nx.connected_components(train_G))
        
        _, matched_train_eg_indices, E_g_train = evaluate_components_tracking(df, train_idx, test_idx, train_comps, 15)
        
        # Now map matched E_g_train back to E_g_true
        # An event in E_g_train is a subset of an event in E_g_true
        for i, true_eg in enumerate(E_g_true):
            true_nodes = set(true_eg["nodes"])
            # Did this station have nodes in this event?
            ant_nodes_in_event = true_nodes.intersection(set(test_idx))
            if len(ant_nodes_in_event) > 0:
                event_tests[i] += 1
                
                # Check if this true event was matched in this fold
                # It was matched if any E_g_train that overlaps with it is in matched_train_eg_indices
                for j_train in matched_train_eg_indices:
                    train_nodes = set(E_g_train[j_train]["nodes"])
                    if len(true_nodes.intersection(train_nodes)) > 0:
                        event_hits[i] += 1
                        break

    # Calculate physics metrics for E_g_true
    profiles = []
    for i, eg in enumerate(E_g_true):
        nodes = eg["nodes"]
        duration = (eg["end"] - eg["start"]) / np.timedelta64(1, 's')
        station_diversity = len(df.loc[nodes, "antenna"].unique())
        temporal_density = len(nodes) / duration if duration > 0 else len(nodes)
        
        # Available Coverage: How many distinct stations were active (had ANY node) during this event window?
        active_stations = df[(df["end_datetime"] >= eg["start"]) & (df["start_datetime"] <= eg["end"])]["antenna"].nunique()
        
        tests = event_tests[i]
        hits = event_hits[i]
        
        if tests == 0:
            continue # Event was not on any of the top 15 stations
            
        recovery_ratio = hits / tests
        
        if recovery_ratio >= 0.8:
            status = "Recovered"
        elif recovery_ratio >= 0.2:
            status = "Partial"
        else:
            status = "Missed"
            
        profiles.append({
            "Event_ID": i,
            "Duration_sec": duration,
            "Station_Diversity": station_diversity,
            "Temporal_Density": temporal_density,
            "Available_Coverage": active_stations,
            "Tests": tests,
            "Hits": hits,
            "Recovery_Ratio": recovery_ratio,
            "Status": status
        })
        
    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(REPORTS_DIR / "struct_event_profiles.csv", index=False)
    
    # ---------------------------------------------------------
    # Block C: Recovery Probability Curve
    # ---------------------------------------------------------
    logging.info("\n=== BLOCK C: Recovery Probability Curve ===")
    
    # By Station Diversity
    div_curve = profiles_df.groupby("Station_Diversity").agg(
        Events=("Event_ID", "count"),
        Recovery_Prob=("Recovery_Ratio", "mean")
    ).reset_index()
    
    # By Available Coverage
    cov_curve = profiles_df.groupby("Available_Coverage").agg(
        Events=("Event_ID", "count"),
        Recovery_Prob=("Recovery_Ratio", "mean")
    ).reset_index()
    
    # Save curves
    with open(REPORTS_DIR / "struct_recovery_curve.csv", "w") as f:
        f.write("--- By Station Diversity ---\n")
        div_curve.to_csv(f, index=False)
        f.write("\n--- By Available Coverage ---\n")
        cov_curve.to_csv(f, index=False)
        
    logging.info("\nDone! Reports saved to event_graph/reports/struct_*.csv")

if __name__ == "__main__":
    main()
