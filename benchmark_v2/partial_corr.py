import pandas as pd
from scipy.stats import spearmanr
import numpy as np

from pathlib import Path
import sys

metrics_path = Path("event_graph/reports/coverage_metrics.csv")
if not metrics_path.exists():
    print("ERROR: event_graph/reports/coverage_metrics.csv not found.")
    print("Please run coverage_analysis.py first.")
    sys.exit(1)

df = pd.read_csv(metrics_path)
def spearman_partial(x, y, z):
    # Calculate rank-based partial correlation
    r_xy, _ = spearmanr(x, y)
    r_xz, _ = spearmanr(x, z)
    r_yz, _ = spearmanr(y, z)
    
    numerator = r_xy - (r_xz * r_yz)
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    r_xy_z = numerator / denominator
    return r_xy_z

overlap = df["temporal_overlap"].values
err = df["ERR"].values
size = df["station_size"].values

partial_r = spearman_partial(overlap, err, size)
print(f"Partial Spearman correlation (ERR, Overlap | Size) = {partial_r:.3f}")
