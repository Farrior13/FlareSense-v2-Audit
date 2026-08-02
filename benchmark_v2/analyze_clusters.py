import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def get_station_bins(counts):
    bins = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, ">10": 0}
    for c in counts:
        if c == 1:
            bins["1"] += 1
        elif c == 2:
            bins["2"] += 1
        elif 3 <= c <= 5:
            bins["3-5"] += 1
        elif 6 <= c <= 10:
            bins["6-10"] += 1
        else:
            bins[">10"] += 1
    return bins

def analyze_strategy(df, strategy_name, group_col):
    total_samples = len(df)
    
    # Calculate durations and station counts per event
    stats = df.groupby(group_col).agg(
        samples=('antenna', 'count'),
        stations=('antenna', 'nunique'),
        start_time=('start_datetime', 'min'),
        end_time=('end_datetime', 'max')
    )
    
    stats['duration_min'] = (stats['end_time'] - stats['start_time']).dt.total_seconds() / 60.0
    
    n_events = len(stats)
    largest_share = (stats['samples'].max() / total_samples) * 100 if n_events > 0 else 0
    
    med_dur = stats['duration_min'].median()
    p95_dur = stats['duration_min'].quantile(0.95)
    max_dur = stats['duration_min'].max()
    
    station_bins = get_station_bins(stats['stations'])
    
    print(f"\n--- Strategy: {strategy_name} ---")
    print(f"Events: {n_events}")
    print(f"Largest Event Share: {largest_share:.2f}% ({stats['samples'].max()} samples)")
    print(f"Duration (min): Median={med_dur:.1f} | 95%={p95_dur:.1f} | Max={max_dur:.1f}")
    print(f"Stations per event: {station_bins}")
    
    return {
        "Strategy": strategy_name,
        "Events": n_events,
        "Largest Share (%)": f"{largest_share:.2f}%",
        "Med Dur": f"{med_dur:.0f}m",
        "95% Dur": f"{p95_dur:.0f}m",
        "Max Dur": f"{max_dur:.0f}m",
        "1 st": station_bins["1"],
        "2 st": station_bins["2"],
        "3-5 st": station_bins["3-5"],
        "6-10 st": station_bins["6-10"],
        ">10 st": station_bins[">10"],
    }

def main():
    logging.info("Loading dataset...")
    ds = load_dataset("i4ds/ecallisto_radio_sunburst")
    
    cols = ["manual_label", "start_datetime", "antenna"]
    df_train = ds["train"].to_pandas()[cols]
    df_val = ds["val"].to_pandas()[cols]
    df_test = ds["test"].to_pandas()[cols]
    
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    df["end_datetime"] = df["start_datetime"] + pd.Timedelta(minutes=15)
    
    # Filter only bursts (the physical events we are clustering)
    bursts = df[df["manual_label"] != 0].copy()
    bursts = bursts.sort_values("start_datetime").reset_index(drop=True)
    
    logging.info(f"Total burst samples: {len(bursts)}")
    
    results = []
    
    # 1. 15-min floor
    bursts["event_15m"] = bursts["start_datetime"].dt.floor("15min")
    results.append(analyze_strategy(bursts, "15 min", "event_15m"))
    
    # 2. 30-min floor
    bursts["event_30m"] = bursts["start_datetime"].dt.floor("30min")
    results.append(analyze_strategy(bursts, "30 min", "event_30m"))
    
    # 3. 60-min floor
    bursts["event_60m"] = bursts["start_datetime"].dt.floor("60min")
    results.append(analyze_strategy(bursts, "60 min", "event_60m"))
    
    # 4. Graph Clustering (Strict Intersection)
    logging.info("\nBuilding interval graph...")
    # NOTE: This graph connects ALL overlapping bursts (including same station)
    # for physical event clustering analysis. This is intentionally different
    # from the benchmark graph (cross_station_only=True) which tests
    # inter-station signal structure.
    G = nx.Graph()
    # Add nodes
    for i in range(len(bursts)):
        G.add_node(i)
        
    # NOTE: This graph connects ALL overlapping bursts (including same station)
    # for physical event clustering analysis. This is intentionally different
    # from the benchmark graph (cross_station_only=True) which tests
    # inter-station signal structure.
    # Sort bursts by start_datetime for efficient overlap checking
    # Since they are 15-min intervals, we only need to check forward a bit
    start_times = bursts["start_datetime"].values
    end_times = bursts["end_datetime"].values
    
    for i in range(len(bursts)):
        for j in range(i+1, len(bursts)):
            # If j starts after i ends, no more overlaps possible (since sorted)
            if start_times[j] >= end_times[i]:
                break
            # Otherwise they overlap
            G.add_edge(i, j)
            
    components = list(nx.connected_components(G))
    comp_map = {}
    for comp_id, nodes in enumerate(components):
        for node in nodes:
            comp_map[node] = comp_id
            
    bursts["event_graph"] = bursts.index.map(comp_map)
    results.append(analyze_strategy(bursts, "Graph", "event_graph"))
    
    # Print summary table
    print("\n\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
