import argparse
import pandas as pd
import numpy as np
import networkx as nx
from datasets import load_dataset
from pathlib import Path
import yaml
import logging
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

logging.basicConfig(level=logging.INFO, format="%(message)s")
REPORTS_DIR = Path("event_graph/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_config(config_path):
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def build_graph_events(df, window_minutes=15):
    """Build events using Strict Temporal Intersection of windows."""
    G = nx.Graph()
    G.add_nodes_from(range(len(df)))
    
    start_times = df["start_datetime"].values
    if "end_datetime" in df.columns:
        end_times = df["end_datetime"].values
    else:
        end_times = (df["start_datetime"] + pd.Timedelta(minutes=window_minutes)).values
    antennas = df["antenna"].values
    
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if start_times[j] >= end_times[i]:
                break
            if antennas[i] != antennas[j]:
                G.add_edge(i, j)
            
    components = list(nx.connected_components(G))
    comp_map = {}
    for comp_id, nodes in enumerate(components):
        for node in nodes:
            comp_map[node] = comp_id
            
    return df.index.map(comp_map), G, components

def evaluate_topology(G, components, total_nodes):
    return {
        "num_components": len(components),
        "largest_component": max([len(c) for c in components]) if components else 0,
        "gcf": max([len(c) for c in components]) / total_nodes if total_nodes > 0 else 0
    }

def evaluate_metrics(df, event_col="event_id"):
    total_samples = len(df)
    active_clusters = df.groupby(event_col)["manual_label"].sum() > 0
    active_event_ids = active_clusters[active_clusters].index
    
    noise_in_active = df[(df[event_col].isin(active_event_ids)) & (df["manual_label"] == 0)]
    nir = len(noise_in_active) / total_samples if total_samples > 0 else 0
    
    purities = []
    for comp_id in active_event_ids:
        comp_data = df[df[event_col] == comp_id]
        dominant_count = comp_data["manual_label"].value_counts().max()
        purities.append(dominant_count / len(comp_data))
    
    mean_purity = np.mean(purities) if purities else 1.0
    return nir, mean_purity

def test_sensitivity(df, windows):
    logging.info("\n--- 1. Temporal Scale Sensitivity ---")
    results = {}
    df_temp = df.copy()
    for w in windows:
        df_temp["end_datetime"] = df_temp["start_datetime"] + pd.Timedelta(minutes=w)
        labels, G, components = build_graph_events(df_temp, w)
        df_temp[f"event_id_{w}"] = labels
        topo = evaluate_topology(G, components, len(df_temp))
        nir, purity = evaluate_metrics(df_temp, f"event_id_{w}")
        
        results[str(w)] = {
            "num_components": topo["num_components"],
            "largest_component": topo["largest_component"],
            "gcf": topo["gcf"],
            "nir": nir,
            "mean_purity": purity
        }
        logging.info(f"Window {w}m: {topo['num_components']} comps, GCF={topo['gcf']:.2%}, Purity={purity:.2%}")
    return results

def test_null_models(df, real_G, window_minutes, seed):
    np.random.seed(seed)
    df_sorted = df.sort_values("start_datetime").reset_index(drop=True)
    time_span = df_sorted["start_datetime"].max() - df_sorted["start_datetime"].min()
    
    logging.info("\n--- 2. Null Models ---")
    results = {}
    
    # Null 1: Circular Time Shift (20 runs)
    logging.info("Running Null 1 (Circular shift 20 runs)...")
    null1_gcfs = []
    null1_purities = []
    for i in range(20):
        null1 = df_sorted.copy()
        for ant in null1["antenna"].unique():
            mask = null1["antenna"] == ant
            shift_seconds = np.random.uniform(0, time_span.total_seconds())
            shifted_times = null1.loc[mask, "start_datetime"] + pd.Timedelta(seconds=shift_seconds)
            overflow_mask = shifted_times > null1["start_datetime"].max()
            shifted_times[overflow_mask] = null1["start_datetime"].min() + (shifted_times[overflow_mask] - null1["start_datetime"].max())
            null1.loc[mask, "start_datetime"] = shifted_times
        null1["end_datetime"] = null1["start_datetime"] + pd.Timedelta(minutes=window_minutes)
        null1 = null1.sort_values("start_datetime").reset_index(drop=True)
        
        n_labels, n_G, n_comps = build_graph_events(null1, window_minutes)
        null1["event_id"] = n_labels
        topo = evaluate_topology(n_G, n_comps, len(null1))
        nir, purity = evaluate_metrics(null1)
        null1_gcfs.append(topo["gcf"])
        null1_purities.append(purity)
        
    results["null1_time_shift"] = {
        "runs": 20,
        "mean_gcf": np.mean(null1_gcfs),
        "std_gcf": np.std(null1_gcfs),
        "mean_purity": np.mean(null1_purities),
        "std_purity": np.std(null1_purities)
    }
    
    # Null 2: Configuration Model (Degree preserving)
    logging.info("Running Null 2 (Configuration Model)...")
    degrees = [d for n, d in real_G.degree()]
    null2_G = nx.configuration_model(degrees, create_using=nx.Graph)
    null2_comps = list(nx.connected_components(null2_G))
    null2_topo = evaluate_topology(null2_G, null2_comps, len(df))
    
    # Degree verification
    degrees_null = [d for n, d in null2_G.degree()]
    mean_abs_deg_error = np.mean(np.abs(np.array(degrees) - np.array(degrees_null)))
    
    results["null2_degree_preserving"] = {
        "topology_metrics": null2_topo,
        "semantic_metrics": None,
        "verification": {
            "mean_abs_degree_error": mean_abs_deg_error,
            "passed": mean_abs_deg_error < 0.1
        }
    }
    
    # Null 3: Station/Time Shuffle
    logging.info("Running Null 3 (Station-time shuffle)...")
    null3 = df_sorted.copy()
    null3["start_datetime"] = null3.groupby("antenna")["start_datetime"].transform(np.random.permutation)
    null3["end_datetime"] = null3["start_datetime"] + pd.Timedelta(minutes=window_minutes)
    null3 = null3.sort_values("start_datetime").reset_index(drop=True)
    
    n_labels, n_G, n_comps = build_graph_events(null3, window_minutes)
    null3["event_id"] = n_labels
    topo = evaluate_topology(n_G, n_comps, len(null3))
    nir, purity = evaluate_metrics(null3)
    
    results["null3_station_shuffle"] = {
        "gcf": topo["gcf"],
        "mean_purity": purity
    }
    
    return results

def test_cross_year(df, window_minutes):
    logging.info("\n--- 3. Cross-year Stability ---")
    years = df['start_datetime'].dt.year
    unique_years = years.unique()
    
    results = {}
    for y in sorted(unique_years):
        b_year = df[years == y].copy().reset_index(drop=True)
        if len(b_year) == 0: continue
        
        labels, G, comps = build_graph_events(b_year, window_minutes)
        b_year['event_id'] = labels
        topo = evaluate_topology(G, comps, len(b_year))
        nir, purity = evaluate_metrics(b_year)
        
        results[str(y)] = {
            "nodes": len(b_year),
            "stations": b_year["antenna"].nunique(),
            "gcf": topo["gcf"],
            "purity": purity
        }
        logging.info(f"{y}: {len(b_year)} nodes, {b_year['antenna'].nunique()} stations | GCF={topo['gcf']:.2%}, Purity={purity:.2%}")
    return results

def test_station_ablation(df, window_minutes, base_purity, base_gcf):
    logging.info("\n--- 4. Station Robustness (LOSO) ---")
    counts = df["antenna"].value_counts()
    top_5 = counts.head(5).index.tolist()
    random_5 = np.random.choice(counts.index, min(5, len(counts.index)), replace=False).tolist()
    
    ablations = list(set(top_5 + random_5))
    results = {}
    
    for st in ablations:
        b_sub = df[df["antenna"] != st].copy().reset_index(drop=True)
        labels, G, comps = build_graph_events(b_sub, window_minutes)
        b_sub['event_id'] = labels
        topo = evaluate_topology(G, comps, len(b_sub))
        nir, purity = evaluate_metrics(b_sub)
        
        delta_p = purity - base_purity
        delta_g = topo["gcf"] - base_gcf
        
        group = "Top-5" if st in top_5 else "Random"
        results[st] = {
            "group": group,
            "delta_purity": delta_p,
            "delta_gcf": delta_g
        }
        logging.info(f"Remove {st} ({group}): Δ Purity={delta_p:+.2%}, Δ GCF={delta_g:+.2%}")
        
    return results

def test_event_isolation(df, window_minutes):
    logging.info("\n--- 5. Event Isolation (Leakage & Forbidden Edges) ---")
    results = {}
    gap_thresholds = [30, 60, 120, 240]
    
    bursts = df[df["manual_label"] != 0].sort_values("start_datetime").reset_index(drop=True)
    if len(bursts) == 0:
        return results
        
    for gap in gap_thresholds:
        # Define pseudo-events separated by 'gap' minutes of global silence among bursts
        diffs = bursts["start_datetime"].diff().dt.total_seconds() / 60.0
        # diffs[0] is NaN, fill with 0
        diffs = diffs.fillna(0)
        
        event_ids = (diffs > gap).cumsum()
        bursts["proxy_event_id"] = event_ids
        
        # Build graph ONLY on bursts to check burst-to-burst connections
        labels, b_G, comps = build_graph_events(bursts, window_minutes)
        
        total_edges = b_G.number_of_edges()
        if total_edges == 0:
            continue
            
        cross_gap_edges = 0
        proxy_ids = bursts["proxy_event_id"].values
        for u, v in b_G.edges():
            if proxy_ids[u] != proxy_ids[v]:
                cross_gap_edges += 1
                
        edge_leakage_rate = cross_gap_edges / total_edges
        
        # Forbidden Edge Rate: out of all pairs that are in different proxy events, how many have an edge?
        # Number of possible cross_gap pairs:
        # Sum of (size(A) * size(B)) for all pairs of events A, B
        sizes = bursts["proxy_event_id"].value_counts().values
        possible_cross_pairs = (np.sum(sizes)**2 - np.sum(sizes**2)) / 2
        
        forbidden_edge_rate = cross_gap_edges / possible_cross_pairs if possible_cross_pairs > 0 else 0
        
        results[f"gap_{gap}m"] = {
            "edge_leakage_rate": edge_leakage_rate,
            "forbidden_edge_rate": forbidden_edge_rate
        }
        logging.info(f"Gap {gap}m: Leakage Rate={edge_leakage_rate:.4f}, Forbidden Rate={forbidden_edge_rate:.6f}")
        
    return results

def run_all(df, window_minutes, config):
    logging.info("Starting comprehensive audit protocol...")
    df = df.sort_values("start_datetime").reset_index(drop=True)
    seed = config.get("seeds", {}).get("null_model", 42)
    np.random.seed(seed)
    
    results_dict = {
        "metadata": {
            "total_samples": len(df),
            "window_minutes_base": window_minutes
        }
    }
    
    # 1. Temporal Scale Sensitivity
    windows = [5, 10, 15, 20, 30, 45, 60]
    sens_res = test_sensitivity(df, windows)
    results_dict["sensitivity_analysis"] = {"window_minutes": sens_res}
    
    # Baseline stats for window_minutes
    base_stats = sens_res[str(window_minutes)]
    base_purity = base_stats["mean_purity"]
    base_gcf = base_stats["gcf"]
    
    # Rebuild base graph for Nulls
    labels, real_G, comps = build_graph_events(df, window_minutes)
    
    # 2. Null Models
    results_dict["null_models"] = test_null_models(df, real_G, window_minutes, seed)
    
    # 3. Cross-year
    results_dict["cross_year"] = test_cross_year(df, window_minutes)
    
    # 4. Station Ablation
    results_dict["station_ablation"] = test_station_ablation(df, window_minutes, base_purity, base_gcf)
    
    # 5. Event Isolation
    results_dict["event_isolation"] = test_event_isolation(df, window_minutes)
    
    # 6. Final Summary Score
    logging.info("\n--- 6. Stress Test Summary ---")
    real_beats_null1 = base_purity > results_dict["null_models"]["null1_time_shift"]["mean_purity"]
    # Check if real graph is structurally different from random degree-preserving graph (e.g. higher clustering, but here we check GCF difference)
    null2_gcf = results_dict["null_models"]["null2_degree_preserving"]["topology_metrics"]["gcf"]
    real_beats_null2 = base_gcf != null2_gcf # They shouldn't be identical if structure matters
    
    station_robust = all(abs(val["delta_purity"]) < 0.05 and abs(val["delta_gcf"]) < 0.02 for val in results_dict["station_ablation"].values())
    
    # Check if leakage is low (e.g. < 1% for 120m gap)
    gap_120 = results_dict.get("event_isolation", {}).get("gap_120m", {})
    event_leakage_low = gap_120.get("edge_leakage_rate", 1.0) < 0.01
    
    # Window stability: purity doesn't collapse at 30 mins
    window_stable = sens_res.get("30", {}).get("mean_purity", 0) > 0.70
    
    summary = {
        "real_beats_null1": bool(real_beats_null1),
        "real_beats_null2": bool(real_beats_null2),
        "station_robust": bool(station_robust),
        "event_leakage_low": bool(event_leakage_low),
        "window_stable": bool(window_stable),
        "audit_passed": bool(real_beats_null1 and station_robust and event_leakage_low and window_stable)
    }
    results_dict["stress_test_summary"] = summary
    logging.info(json.dumps(summary, indent=2))
    
    with open("benchmark_results_v2.json", "w") as f:
        json.dump(results_dict, f, indent=2, cls=NpEncoder)
    logging.info("\nFull results saved to benchmark_results_v2.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "all"], default="test")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--run-all", action="store_true", help="Run the full comprehensive audit suite")
    args = parser.parse_args()
    
    config = load_config(args.config)
    dataset_name = config.get("benchmark", {}).get("dataset", "i4ds/ecallisto_radio_sunburst")
    window_minutes = config.get("benchmark", {}).get("window_minutes", 15)
    
    split_mode = args.split if args.split else config.get("benchmark", {}).get("split", "test")
    
    ds = load_dataset(dataset_name)
    cols = ["manual_label", "start_datetime", "antenna"]
    
    if split_mode == "all":
        df = pd.concat([ds[s].to_pandas()[cols] for s in ['train', 'val', 'test']], ignore_index=True)
    else:
        df = ds["test"].to_pandas()[cols]
        
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    if "end_datetime" not in df.columns:
        df["end_datetime"] = df["start_datetime"] + pd.Timedelta(minutes=window_minutes)
        
    if args.run_all:
        run_all(df, window_minutes, config)
    else:
        # Fallback to basic graph if not running full audit
        df = df.sort_values("start_datetime").reset_index(drop=True)
        labels, G, comps = build_graph_events(df, window_minutes)
        df["event_id"] = labels
        nir, purity = evaluate_metrics(df, event_col="event_id")
        print(f"Basic run complete. Purity: {purity:.2%}. Use --run-all for full audit.")

if __name__ == "__main__":
    main()
