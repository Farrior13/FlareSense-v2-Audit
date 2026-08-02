import pandas as pd
import numpy as np
from datasets import load_dataset
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

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
    
    bursts = df[df["manual_label"] != 0].copy()
    bursts = bursts.sort_values("start_datetime").reset_index(drop=True)
    
    logging.info("Building interval graph...")
    G = nx.Graph()
    for i in range(len(bursts)):
        G.add_node(i)
        
    start_times = bursts["start_datetime"].values
    end_times = bursts["end_datetime"].values
    
    # NOTE: This graph connects ALL overlapping bursts (including same station)
    # for physical event verification. This is intentionally different from
    # the benchmark graph (cross_station_only=True).
    for i in range(len(bursts)):
        for j in range(i+1, len(bursts)):
            if start_times[j] >= end_times[i]:
                break
            G.add_edge(i, j)
            
    components = list(nx.connected_components(G))
    comp_map = {}
    for comp_id, nodes in enumerate(components):
        for node in nodes:
            comp_map[node] = comp_id
            
    bursts["event_id"] = bursts.index.map(comp_map)
    
    # Calculate stats
    stats = bursts.groupby("event_id").agg(
        samples=('antenna', 'count'),
        stations=('antenna', 'nunique'),
        start_time=('start_datetime', 'min'),
        end_time=('end_datetime', 'max')
    )
    stats['duration_min'] = (stats['end_time'] - stats['start_time']).dt.total_seconds() / 60.0
    
    print("\n--- 1. Top 20 Largest Events ---")
    top20 = stats.sort_values("samples", ascending=False).head(20)
    top20_print = top20.copy()
    top20_print['start_time'] = top20_print['start_time'].dt.strftime('%Y-%m-%d %H:%M')
    top20_print['end_time'] = top20_print['end_time'].dt.strftime('%Y-%m-%d %H:%M')
    print(top20_print.to_string())
    
    print("\n--- 2. Duration Tail Distribution (minutes) ---")
    quantiles = [0.50, 0.75, 0.90, 0.95, 0.99]
    tail = stats['duration_min'].quantile(quantiles)
    for q, val in tail.items():
        print(f"{int(q*100)}%: {val:.1f} min")
    print(f"Max: {stats['duration_min'].max():.1f} min")
    
    print("\n--- 3. Random 5 Events Inspection ---")
    np.random.seed(42)
    random_events = np.random.choice(stats.index, size=5, replace=False)
    for ev in random_events:
        print(f"\nEvent ID: {ev}")
        ev_data = bursts[bursts["event_id"] == ev]
        print(f"Total samples: {len(ev_data)}, Unique stations: {ev_data['antenna'].nunique()}")
        print(f"Global Start: {ev_data['start_datetime'].min()} | Global End: {ev_data['end_datetime'].max()}")
        # print first 3 samples
        print(ev_data[['start_datetime', 'antenna']].head(3).to_string(index=False))
        if len(ev_data) > 3:
            print("...")

if __name__ == "__main__":
    main()
