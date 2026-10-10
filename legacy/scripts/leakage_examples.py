#!/usr/bin/env python3
"""Find and display concrete examples of event-level leakage between train and test.

Loads the HuggingFace dataset, identifies shared 15-min time buckets between
train+val burst samples and test burst samples, and reports detailed examples.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train+val and test splits from HuggingFace."""
    cols = ["manual_label", "prob", "model_label", "start_datetime", "antenna"]
    
    print("Loading dataset from HuggingFace...")
    df_train = load_dataset("i4ds/ecallisto_radio_sunburst", split="train").select_columns(cols).to_pandas()
    df_val = load_dataset("i4ds/ecallisto_radio_sunburst", split="val").select_columns(cols).to_pandas()
    df_test = load_dataset("i4ds/ecallisto_radio_sunburst", split="test").select_columns(cols).to_pandas()
    
    df_tv = pd.concat([df_train, df_val], ignore_index=True)
    df_tv["start_datetime"] = pd.to_datetime(df_tv["start_datetime"])
    df_test["start_datetime"] = pd.to_datetime(df_test["start_datetime"])
    
    return df_tv, df_test


def find_shared_buckets(df_tv: pd.DataFrame, df_test: pd.DataFrame, freq: str = "15min"):
    """Find time buckets shared between train+val and test burst samples."""
    tv_burst = df_tv[df_tv["manual_label"] != 0].copy()
    test_burst = df_test[df_test["manual_label"] != 0].copy()
    
    tv_burst["bucket"] = tv_burst["start_datetime"].dt.floor(freq)
    test_burst["bucket"] = test_burst["start_datetime"].dt.floor(freq)
    
    shared = set(tv_burst["bucket"].unique()) & set(test_burst["bucket"].unique())
    return tv_burst, test_burst, shared


def build_examples(tv_burst: pd.DataFrame, test_burst: pd.DataFrame, shared_buckets: set) -> list[dict]:
    """Build structured examples from shared event buckets."""
    examples = []
    for bucket in sorted(shared_buckets):
        tv_in = tv_burst[tv_burst["bucket"] == bucket]
        te_in = test_burst[test_burst["bucket"] == bucket]
        
        examples.append({
            "bucket": str(bucket),
            "train_stations": sorted(tv_in["antenna"].unique().tolist()),
            "test_stations": sorted(te_in["antenna"].unique().tolist()),
            "train_n": len(tv_in),
            "test_n": len(te_in),
            "n_train_stations": len(tv_in["antenna"].unique()),
            "n_test_stations": len(te_in["antenna"].unique()),
        })
    
    return sorted(examples, key=lambda x: x["test_n"], reverse=True)


def print_top_events(examples: list[dict], n: int = 15):
    """Print top N leaked events by test sample count."""
    print(f"\n{'='*75}")
    print(f"TOP {n} LEAKED EVENTS (by test samples)")
    print(f"{'='*75}")
    print(f"{'Bucket':<22} {'Train St':>8} {'Test St':>7} {'Train N':>7} {'Test N':>6}")
    print("-" * 55)
    for ex in examples[:n]:
        print(f"{ex['bucket']:<22} {ex['n_train_stations']:>8} "
              f"{ex['n_test_stations']:>7} {ex['train_n']:>7} {ex['test_n']:>6}")


def print_detailed_examples(tv_burst: pd.DataFrame, test_burst: pd.DataFrame,
                            examples: list[dict], n: int = 5):
    """Print detailed station-level breakdown for top N examples."""
    print(f"\n{'='*75}")
    print(f"DETAILED LEAKAGE EXAMPLES (top {n})")
    print(f"{'='*75}")
    
    for i, ex in enumerate(examples[:n]):
        bucket = pd.Timestamp(ex["bucket"])
        print(f"\n--- Example {i+1}: {bucket} ---")
        
        tv_in = tv_burst[tv_burst["bucket"] == bucket]
        te_in = test_burst[test_burst["bucket"] == bucket]
        
        print(f"  TRAIN+VAL ({len(tv_in)} samples from {ex['n_train_stations']} stations):")
        for st in sorted(tv_in["antenna"].unique()):
            st_data = tv_in[tv_in["antenna"] == st]
            times = sorted(st_data["start_datetime"].dt.strftime("%H:%M").unique())
            print(f"    {st}: {len(st_data)} samples, times={times}")
        
        print(f"  TEST ({len(te_in)} samples from {ex['n_test_stations']} stations):")
        for st in sorted(te_in["antenna"].unique()):
            st_data = te_in[te_in["antenna"] == st]
            times = sorted(st_data["start_datetime"].dt.strftime("%H:%M").unique())
            mean_prob = st_data["prob"].mean()
            preds = st_data["model_label"].values.tolist()
            correct = "✓" if all(p == 1 for p in preds) else "✗" if all(p == 0 for p in preds) else "~"
            print(f"    {st}: {len(st_data)} samples, times={times}, "
                  f"prob={mean_prob:.3f}, pred={preds} {correct}")


def print_structure_stats(examples: list[dict]):
    """Print overlap structure statistics."""
    n_test_stations = [ex["n_test_stations"] for ex in examples]
    n_train_stations = [ex["n_train_stations"] for ex in examples]
    
    print(f"\n{'='*75}")
    print("OVERLAP STRUCTURE STATISTICS")
    print(f"{'='*75}")
    print(f"Total shared event buckets: {len(examples)}")
    print(f"Events with 1 test station:  {sum(1 for n in n_test_stations if n == 1)}")
    print(f"Events with 2 test stations: {sum(1 for n in n_test_stations if n == 2)}")
    print(f"Events with 3+ test stations: {sum(1 for n in n_test_stations if n >= 3)}")
    print(f"Events with 5+ test stations: {sum(1 for n in n_test_stations if n >= 5)}")
    print(f"Median train stations per event: {np.median(n_train_stations):.0f}")
    print(f"Median test stations per event:  {np.median(n_test_stations):.0f}")


def main():
    parser = argparse.ArgumentParser(description="Find concrete event-level leakage examples")
    parser.add_argument("--output-dir", default="results", help="Output directory for JSON results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df_tv, df_test = load_data()
    tv_burst, test_burst, shared = find_shared_buckets(df_tv, df_test, "15min")
    examples = build_examples(tv_burst, test_burst, shared)
    
    print_top_events(examples)
    print_detailed_examples(tv_burst, test_burst, examples)
    print_structure_stats(examples)
    
    # Save to JSON
    output_path = Path(args.output_dir) / "leakage_examples.json"
    with open(output_path, "w") as f:
        json.dump({
            "total_shared_buckets": len(examples),
            "top_15_events": examples[:15],
            "structure": {
                "events_1_test_station": sum(1 for ex in examples if ex["n_test_stations"] == 1),
                "events_2_test_stations": sum(1 for ex in examples if ex["n_test_stations"] == 2),
                "events_3plus_test_stations": sum(1 for ex in examples if ex["n_test_stations"] >= 3),
                "events_5plus_test_stations": sum(1 for ex in examples if ex["n_test_stations"] >= 5),
            }
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
