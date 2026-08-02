import pandas as pd
import numpy as np
import networkx as nx
from datasets import load_dataset
from pathlib import Path
import logging
from tqdm import tqdm
from structural_core import build_weighted_graph, get_pruned_graph, run_targeted_removal_eval, evaluate_components_tracking
from eg_loso_core import extract_global_events

logging.basicConfig(level=logging.INFO, format="%(message)s")
REPORTS_DIR = Path("event_graph/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def apply_circular_time_shift(df, seed):
    dataset_start = df["start_datetime"].min()
    dataset_end = df["start_datetime"].max()
    dataset_duration = dataset_end - dataset_start
    
    np.random.seed(seed)
    df_null = df.copy()
    
    for ant in df_null["antenna"].unique():
        offset = np.random.uniform(0, dataset_duration.total_seconds())
        mask = df_null["antenna"] == ant
        new_time = (df_null.loc[mask, "start_datetime"] - dataset_start + pd.Timedelta(seconds=offset)) % dataset_duration
        df_null.loc[mask, "start_datetime"] = dataset_start + new_time
        
    df_null["end_datetime"] = df_null["start_datetime"] + pd.Timedelta(minutes=15)
    df_null = df_null.sort_values("start_datetime").reset_index(drop=True)
    return df_null

def main():
    logging.info("Loading dataset...")
    ds = load_dataset("i4ds/ecallisto_radio_sunburst", split="test")
    cols = ["manual_label", "start_datetime", "antenna"]
    df = ds.to_pandas()[cols]
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    df["end_datetime"] = df["start_datetime"] + pd.Timedelta(minutes=15)
    df = df.sort_values("start_datetime").reset_index(drop=True)
    
    top_stations = df[df["manual_label"]==1]["antenna"].value_counts().head(15).index
    
    # Base Graph count
    G_base = build_weighted_graph(df, 15)
    base_edges = G_base.number_of_edges()
    logging.info(f"Base Graph Edges: {base_edges}")
    
    N_SEEDS = 20
    
    null_edges_list = []
    base_err_list = []
    weak_err_list = []
    strong_err_list = []
    
    # For curves, we collect rows of (Seed, Station_Diversity, Recovery_Prob)
    curve_rows = []
    
    logging.info(f"Running {N_SEEDS} CT-Null shifts...")
    for seed in tqdm(range(N_SEEDS), desc="CT-Null Seeds"):
        df_null = apply_circular_time_shift(df, seed=42+seed)
        G_null = build_weighted_graph(df_null, 15)
        null_edges_list.append(G_null.number_of_edges())
        
        # Pruning EVAL
        base_err = run_targeted_removal_eval(df_null, G_null, top_stations, 15)
        base_err_list.append(base_err)
        
        G_weak = get_pruned_graph(G_null, "weak", 0.3, betweenness=None)
        weak_err = run_targeted_removal_eval(df_null, G_weak, top_stations, 15)
        weak_err_list.append(weak_err)
        
        G_strong = get_pruned_graph(G_null, "strong", 0.3, betweenness=None)
        strong_err = run_targeted_removal_eval(df_null, G_strong, top_stations, 15)
        strong_err_list.append(strong_err)
        
        # Recovery Curve EVAL
        full_comps = list(nx.connected_components(G_null))
        E_g_all = extract_global_events(df_null, full_comps)
        E_g_true = [e for e in E_g_all if e["label"] == 1]
        
        event_hits = {i: 0 for i in range(len(E_g_true))}
        event_tests = {i: 0 for i in range(len(E_g_true))}
        
        for ant in top_stations:
            test_idx = df_null[df_null["antenna"] == ant].index
            train_idx = df_null.index.difference(test_idx)
            train_G = G_null.subgraph(train_idx)
            train_comps = list(nx.connected_components(train_G))
            
            _, matched_train_eg_indices, E_g_train = evaluate_components_tracking(df_null, train_idx, test_idx, train_comps, 15)
            
            for i, true_eg in enumerate(E_g_true):
                true_nodes = set(true_eg["nodes"])
                ant_nodes_in_event = true_nodes.intersection(set(test_idx))
                if len(ant_nodes_in_event) > 0:
                    event_tests[i] += 1
                    for j_train in matched_train_eg_indices:
                        train_nodes = set(E_g_train[j_train]["nodes"])
                        if len(true_nodes.intersection(train_nodes)) > 0:
                            event_hits[i] += 1
                            break
                            
        profiles = []
        for i, eg in enumerate(E_g_true):
            nodes = eg["nodes"]
            station_diversity = len(df_null.loc[nodes, "antenna"].unique())
            tests = event_tests[i]
            hits = event_hits[i]
            
            if tests == 0: continue
            recovery_ratio = hits / tests
            
            profiles.append({
                "Station_Diversity": station_diversity,
                "Recovery_Ratio": recovery_ratio
            })
            
        profiles_df = pd.DataFrame(profiles)
        if len(profiles_df) > 0:
            div_curve = profiles_df.groupby("Station_Diversity").agg(
                Recovery_Prob=("Recovery_Ratio", "mean")
            ).reset_index()
            div_curve["Seed"] = seed
            curve_rows.extend(div_curve.to_dict('records'))

    # ---------------------------------------------------------
    # Aggregating and Saving Results
    # ---------------------------------------------------------
    
    # 1. Edge Pruning
    pruning_results = [
        {"Strategy": "Baseline", "DropRate": 0.0, "ERR_mean": np.mean(base_err_list), "ERR_std": np.std(base_err_list)},
        {"Strategy": "weak", "DropRate": 0.3, "ERR_mean": np.mean(weak_err_list), "ERR_std": np.std(weak_err_list)},
        {"Strategy": "strong", "DropRate": 0.3, "ERR_mean": np.mean(strong_err_list), "ERR_std": np.std(strong_err_list)}
    ]
    pd.DataFrame(pruning_results).to_csv(REPORTS_DIR / "null_temporal_pruning.csv", index=False)
    
    # 2. Recovery Curve
    curve_df_all = pd.DataFrame(curve_rows)
    final_curve = curve_df_all.groupby("Station_Diversity").agg(
        Recovery_Prob_mean=("Recovery_Prob", "mean"),
        Recovery_Prob_std=("Recovery_Prob", "std")
    ).reset_index()
    # fill NaN std with 0 for bins that only appeared in 1 seed (or set to 0)
    final_curve["Recovery_Prob_std"] = final_curve["Recovery_Prob_std"].fillna(0)
    final_curve.to_csv(REPORTS_DIR / "null_temporal_curve.csv", index=False)
    
    # 3. Summary
    with open(REPORTS_DIR / "null_temporal_summary.txt", "w") as f:
        f.write(f"Real Edges: {base_edges}\n")
        f.write(f"Null Edges (mean +- std): {np.mean(null_edges_list):.1f} +- {np.std(null_edges_list):.1f}\n")
        
    logging.info("Done! Results saved to event_graph/reports/null_temporal_*")

if __name__ == "__main__":
    main()
