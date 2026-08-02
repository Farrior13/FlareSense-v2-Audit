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

def evaluate_components(df, train_idx, test_idx, train_comps, window_minutes=15, iot_threshold=0.5):
    assert len(train_idx.intersection(test_idx)) == 0, "LEAKAGE: train/test overlap!"
    # Use train-only df to prevent accidental test-label leakage
    train_df = df.loc[train_idx]
    E_g = extract_global_events(train_df, train_comps)
    E_g_bursts = [e for e in E_g if e["label"] == 1]
    
    test_antennas = df.loc[test_idx, "antenna"].unique()
    metrics = []
    
    for ant in test_antennas:
        ant_test_idx = df.loc[test_idx][df.loc[test_idx, "antenna"] == ant].index
        E_h = extract_station_events(df, ant_test_idx, window_minutes)
        if len(E_h) == 0: continue
            
        recovered_events = 0
        for eh in E_h:
            overlaps = []
            for eg in E_g_bursts:
                iot = compute_temporal_iou(eh["start"], eh["end"], eg["start"], eg["end"])
                if iot > iot_threshold:
                    overlaps.append(eg)
            if len(overlaps) > 0:
                recovered_events += 1
                            
        err = recovered_events / len(E_h)
        
        test_burst_nodes = set(df.loc[ant_test_idx][df.loc[ant_test_idx, "manual_label"] == 1].index)
        test_noise_nodes = set(df.loc[ant_test_idx][df.loc[ant_test_idx, "manual_label"] == 0].index)
        
        matched_test_bursts = 0
        matched_test_noise = 0
        for node in ant_test_idx:
            n_s, n_e = df.loc[node, "start_datetime"], df.loc[node, "end_datetime"]
            matched = False
            for eg in E_g_bursts:
                if compute_temporal_iou(n_s, n_e, eg["start"], eg["end"]) > 0:
                    matched = True
                    break
            if matched:
                if node in test_burst_nodes:
                    matched_test_bursts += 1
                else:
                    matched_test_noise += 1
                    
        total_matched = matched_test_bursts + matched_test_noise
        purity = matched_test_bursts / total_matched if total_matched > 0 else 0
        
        metrics.append({
            "ERR": err,
            "MatchPurity": purity,
        })
        
    if not metrics: return None
    return pd.DataFrame(metrics).mean(numeric_only=True).to_dict()

def compute_ucr(df, components):
    if not components: return 0.0
    false_clusters = sum(1 for comp in components if df.loc[list(comp), "manual_label"].sum() == 0)
    return false_clusters / len(components)

def get_topology_metrics(df, G):
    components = list(nx.connected_components(G))
    if not components: return {"GCF": 0, "ClusterCount": 0, "MeanClusterSize": 0, "UCR": 0, "components": []}
    
    gcf = max(len(c) for c in components) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    cluster_count = len(components)
    mean_cluster_size = np.mean([len(c) for c in components])
    ucr = compute_ucr(df, components)
    
    return {
        "GCF": gcf,
        "ClusterCount": cluster_count,
        "MeanClusterSize": mean_cluster_size,
        "UCR": ucr,
        "components": components
    }

def run_eg_loso_on_graph(df, G, top_stations, window_minutes=15):
    errs, purities = [], []
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
        metrics = evaluate_components(df, train_idx, test_idx, train_comps, window_minutes)
        if metrics:
            errs.append(metrics["ERR"])
            purities.append(metrics["MatchPurity"])
    return np.mean(errs), np.mean(purities)

def generate_degree_null(G):
    G_null = G.copy()
    try:
        nx.double_edge_swap(G_null, nswap=len(G_null.edges)*2, max_tries=len(G_null.edges)*10)
    except nx.NetworkXError:
        pass # If swap fails, just return as is or best effort
    return G_null

def generate_er_null(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    G_er = nx.gnm_random_graph(n, m)
    # Map nodes back to original indices
    mapping = dict(zip(G_er.nodes(), G.nodes()))
    return nx.relabel_nodes(G_er, mapping)

def perturb_graph(G, p):
    G_pert = G.copy()
    edges_to_remove = [e for e in G.edges() if np.random.rand() < p]
    G_pert.remove_edges_from(edges_to_remove)
    return G_pert

def main():
    logging.info("Loading dataset...")
    ds = load_dataset("i4ds/ecallisto_radio_sunburst", split="test")
    cols = ["manual_label", "start_datetime", "antenna"]
    df = ds.to_pandas()[cols]
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    df = df.sort_values("start_datetime").reset_index(drop=True)
    
    top_stations = df[df["manual_label"]==1]["antenna"].value_counts().head(15).index
    
    # ---------------------------------------------------------
    # Block A: Null Models
    # ---------------------------------------------------------
    logging.info("\n=== BLOCK A: Null Models ===")
    df_15 = df.copy()
    df_15["end_datetime"] = df_15["start_datetime"] + pd.Timedelta(minutes=15)
    G_sti, _ = build_graph(df_15, 15)
    
    results_null = []
    
    logging.info("Evaluating STI...")
    topo_sti = get_topology_metrics(df_15, G_sti)
    err_sti, pur_sti = run_eg_loso_on_graph(df_15, G_sti, top_stations, 15)
    results_null.append({"Model": "STI", "ERR": err_sti, "MatchPurity": pur_sti, **{k:v for k,v in topo_sti.items() if k!="components"}})
    
    logging.info("Evaluating Degree-Preserving Null...")
    G_deg = generate_degree_null(G_sti)
    topo_deg = get_topology_metrics(df_15, G_deg)
    err_deg, pur_deg = run_eg_loso_on_graph(df_15, G_deg, top_stations, 15)
    results_null.append({"Model": "Degree-Null", "ERR": err_deg, "MatchPurity": pur_deg, **{k:v for k,v in topo_deg.items() if k!="components"}})
    
    logging.info("Evaluating ER Null...")
    G_er = generate_er_null(G_sti)
    topo_er = get_topology_metrics(df_15, G_er)
    err_er, pur_er = run_eg_loso_on_graph(df_15, G_er, top_stations, 15)
    results_null.append({"Model": "Erdos-Renyi", "ERR": err_er, "MatchPurity": pur_er, **{k:v for k,v in topo_er.items() if k!="components"}})
    
    pd.DataFrame(results_null).to_csv(REPORTS_DIR / "adv_null_models.csv", index=False)
    
    # ---------------------------------------------------------
    # Block B: Window Sweep
    # ---------------------------------------------------------
    logging.info("\n=== BLOCK B: Window Sweep ===")
    results_window = []
    windows = [5, 10, 15, 30, 60, 120]
    for w in tqdm(windows, desc="Windows"):
        df_w = df.copy()
        df_w["end_datetime"] = df_w["start_datetime"] + pd.Timedelta(minutes=w)
        G_w, _ = build_graph(df_w, w)
        topo_w = get_topology_metrics(df_w, G_w)
        err_w, pur_w = run_eg_loso_on_graph(df_w, G_w, top_stations, w)
        results_window.append({"Window": w, "ERR": err_w, "MatchPurity": pur_w, **{k:v for k,v in topo_w.items() if k!="components"}})
        
    pd.DataFrame(results_window).to_csv(REPORTS_DIR / "adv_window_sweep.csv", index=False)
    
    # ---------------------------------------------------------
    # Block C: Edge Perturbation
    # ---------------------------------------------------------
    logging.info("\n=== BLOCK C: Edge Perturbation ===")
    results_edge = []
    drop_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    
    for p in tqdm(drop_rates, desc="Edge Drops"):
        errs_p, pur_p = [], []
        n_seeds = 1 if p == 0.0 else 5
        for seed in range(n_seeds):
            np.random.seed(42 + seed)
            G_p = perturb_graph(G_sti, p)
            err, pur = run_eg_loso_on_graph(df_15, G_p, top_stations, 15)
            errs_p.append(err)
            pur_p.append(pur)
            
        results_edge.append({
            "DropRate": p,
            "ERR_mean": np.mean(errs_p),
            "ERR_std": np.std(errs_p),
            "MatchPurity_mean": np.mean(pur_p),
            "MatchPurity_std": np.std(pur_p)
        })
        
    pd.DataFrame(results_edge).to_csv(REPORTS_DIR / "adv_edge_perturbation.csv", index=False)
    logging.info("\nDone! Reports saved to event_graph/reports/adv_*.csv")

if __name__ == "__main__":
    main()
