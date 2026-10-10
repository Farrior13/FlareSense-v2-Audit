import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

RESULTS_DIR = r"C:\Users\User\Desktop\FlareSense-v2-Audit\results"
FIG_DIR = r"C:\Users\User\Desktop\FlareSense-v2-Audit\figures"
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "core_results.json"), "r") as f:
    core = json.load(f)

with open(os.path.join(RESULTS_DIR, "breakdown_results.json"), "r") as f:
    breakdown = json.load(f)

# ==========================================
# FIGURE 1: Leaked vs Clean Performance Collapse
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

metrics = ['Recall', 'F1-Score', 'Precision*']
leaked_vals = [
    core['full_vs_clean']['FlareSense-v2']['bucket_15min']['recall_leaked'] * 100,
    core['paper_reproduction']['FlareSense-v2']['micro']['f1'] * 100,
    core['paper_reproduction']['FlareSense-v2']['micro']['precision'] * 100
]
clean_vals = [
    core['full_vs_clean']['FlareSense-v2']['bucket_15min']['clean']['recall'] * 100,
    core['full_vs_clean']['FlareSense-v2']['bucket_15min']['clean']['f1'] * 100,
    core['full_vs_clean']['FlareSense-v2']['bucket_15min']['clean_precision_at_full_prevalence'] * 100
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, leaked_vals, width, label='Leaked (Observed in Train by >=1 Station)', color='#e05252', edgecolor='#b33939', alpha=0.9)
bars2 = ax1.bar(x + width/2, clean_vals, width, label='Clean (True Unseen Events, Exposure = 0)', color='#4a90e2', edgecolor='#2c6bb5', alpha=0.9)

ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('A: Observational Performance Disparity on Unseen Events', fontsize=11.5, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11, fontweight='semibold')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', alpha=0.6)
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#b33939')

for bar in bars2:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c6bb5')

# Dynamic annotation of drop
drop_pp = leaked_vals[0] - clean_vals[0]
ax1.annotate(f'-{drop_pp:.1f} pp', xy=(0, clean_vals[0]), xytext=(0, (leaked_vals[0] + clean_vals[0]) / 2 + 8),
            ha='center', fontsize=11, fontweight='heavy', color='#d32f2f',
            arrowprops=dict(arrowstyle='->', lw=2, color='#d32f2f'))

# Panel B: Predicted probability distribution quantiles
quants = ['p10', 'p25', 'median', 'p75']
p_leaked = [core['full_vs_clean']['FlareSense-v2']['bucket_15min']['prob_leaked'][q] for q in quants]
p_clean = [core['full_vs_clean']['FlareSense-v2']['bucket_15min']['prob_clean'][q] for q in quants]

ax2.plot(quants, p_leaked, marker='o', lw=2.5, markersize=8, color='#e05252', label=f'Leaked Events (Median = {p_leaked[2]:.3f})')
ax2.plot(quants, p_clean, marker='s', lw=2.5, markersize=8, color='#4a90e2', label=f'Clean Events (Median = {p_clean[2]:.3f})')
ax2.fill_between(quants, p_clean, p_leaked, color='#f5c6cb', alpha=0.35, label='Confidence Disparity Gap')

ax2.axhline(0.426, color='black', linestyle=':', lw=1.5, label='Decision Threshold (0.426)')
ax2.set_ylabel('Predicted Probability P(Burst)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Quantiles of Model Confidence', fontsize=12, fontweight='bold')
ax2.set_title('B: Confidence Shift Between Leaked and Clean Events', fontsize=11.5, fontweight='bold', pad=12)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, alpha=0.6)
ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

PAPER_FIG_DIR = r"C:\Users\User\Desktop\FlareSense-v2-Audit\paper\figures"

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_leakage_disparity.png"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "fig1_leakage_disparity.pdf"))
if os.path.exists(PAPER_FIG_DIR):
    fig.savefig(os.path.join(PAPER_FIG_DIR, "fig1_leakage_disparity.png"), dpi=300)
    fig.savefig(os.path.join(PAPER_FIG_DIR, "fig1_leakage_disparity.pdf"))
plt.close(fig)
print("Saved fig1_leakage_disparity dynamically")

# ==========================================
# FIGURE 2: Exposure-Response Gradient & DiD
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

gradient = core['confound_controls']['FlareSense-v2']['exposure_gradient']
stations_cat = [g['trainval_stations'] for g in gradient]
recalls = [g['recall'] * 100 for g in gradient]
mean_probs = [g['mean_prob'] * 100 for g in gradient]
counts = [g['n'] for g in gradient]

color1 = '#2b5c8f'
ax1.plot(stations_cat, recalls, marker='o', markersize=9, lw=2.8, color=color1, label='Recall (%)')
ax1.plot(stations_cat, mean_probs, marker='^', markersize=8, lw=2.2, linestyle='--', color='#e67e22', label='Mean Predicted Prob (%)')

for i, (cat, rec, n) in enumerate(zip(stations_cat, recalls, counts)):
    ax1.annotate(f"{rec:.1f}%\n(n={n})", (i, rec + 2.5), ha='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Number of Training Stations Observing the Same Burst', fontsize=11, fontweight='bold')
ax1.set_ylabel('Model Performance / Confidence (%)', fontsize=11, fontweight='bold')
ax1.set_title('A: Monotonic Exposure-Response Curve', fontsize=12, fontweight='bold', pad=12)
ax1.set_ylim(25, 108)
ax1.grid(True, alpha=0.6)
ax1.legend(loc='lower right', frameon=True, facecolor='white')

# Panel B: Difference-in-Differences
models = ['Baseline Control\n(Old HF Preds)', 'FlareSense-v2\n(arXiv:2607.26014)']
clean_r = [
    core['full_vs_clean']['old_HF_predictions']['bucket_15min']['clean']['recall'] * 100,
    core['full_vs_clean']['FlareSense-v2']['bucket_15min']['clean']['recall'] * 100
]
leaked_r = [
    core['full_vs_clean']['old_HF_predictions']['bucket_15min']['recall_leaked'] * 100,
    core['full_vs_clean']['FlareSense-v2']['bucket_15min']['recall_leaked'] * 100
]

bx = np.arange(len(models))
bwidth = 0.32
ax2.bar(bx - bwidth/2, clean_r, bwidth, label='Clean Recall', color='#5dade2', edgecolor='#2980b9')
ax2.bar(bx + bwidth/2, leaked_r, bwidth, label='Leaked Recall', color='#e74c3c', edgecolor='#c0392b')

ax2.set_xticks(bx)
ax2.set_xticklabels(models, fontsize=11, fontweight='bold')
ax2.set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
ax2.set_title('B: Difference-in-Differences Causal Isolation', fontsize=12, fontweight='bold', pad=12)
ax2.set_ylim(0, 110)
ax2.grid(axis='y', alpha=0.6)
ax2.legend(loc='upper left', frameon=True, facecolor='white')

# DiD Bracket
did_val = core['confound_controls']['DiD_v2_minus_old_leaked_minus_clean_pp']
ci = core['confound_controls']['DiD_ci95_pp_cluster']
ax2.annotate(f"Causal DiD Boost:\n+{did_val:.1f} pp (p < 0.001)\n95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]",
            xy=(1 + bwidth/2, 90), xytext=(0.5, 95),
            ha='center', fontsize=10, fontweight='heavy', color='#8e44ad',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f4ecf7', edgecolor='#8e44ad', lw=1.5),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='#8e44ad'))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_exposure_and_did.png"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "fig2_exposure_and_did.pdf"))
if os.path.exists(PAPER_FIG_DIR):
    fig.savefig(os.path.join(PAPER_FIG_DIR, "fig2_exposure_and_did.png"), dpi=300)
    fig.savefig(os.path.join(PAPER_FIG_DIR, "fig2_exposure_and_did.pdf"))
plt.close(fig)
print("Saved fig2_exposure_and_did")

# ==========================================
# FIGURE 3: Per-Station Recall Collapse
# ==========================================
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

stations_data = breakdown['FlareSense-v2']['per_station']
# Filter stations with at least 5 clean bursts to have meaningful statistics
valid_stations = [s for s in stations_data if s.get('bursts_clean', 0) >= 5]
# Sort by recall collapse (leaked - clean)
valid_stations.sort(key=lambda x: (x['recall_leaked'] - x['recall_clean']), reverse=True)
top_stations = valid_stations[:10]

y_names = [s['antenna'] for s in top_stations]
rec_leaked = [s['recall_leaked'] * 100 for s in top_stations]
rec_clean = [s['recall_clean'] * 100 for s in top_stations]

y_pos = np.arange(len(y_names))
bar_h = 0.38

rects1 = ax.barh(y_pos - bar_h/2, rec_leaked, bar_h, label='Leaked Recall (Event in Train)', color='#e05252', alpha=0.9)
rects2 = ax.barh(y_pos + bar_h/2, rec_clean, bar_h, label='Clean Recall (Unseen Event)', color='#4a90e2', alpha=0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels(y_names, fontsize=10, fontweight='bold')
ax.invert_yaxis()  # top-down
ax.set_xlabel('Recall (%)', fontsize=11, fontweight='bold')
ax.set_title('Station-Level Recall Degradation (Top 10 Stations with >=5 Clean Bursts)', fontsize=12, fontweight='bold', pad=14)
ax.set_xlim(0, 115)
ax.grid(axis='x', alpha=0.6)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95)

for i, (r_l, r_c) in enumerate(zip(rec_leaked, rec_clean)):
    drop = r_l - r_c
    ax.text(r_l + 1.5, i - bar_h/2, f"{r_l:.1f}%", va='center', fontsize=9, fontweight='semibold', color='#b33939')
    ax.text(r_c + 1.5, i + bar_h/2, f"{r_c:.1f}%", va='center', fontsize=9, fontweight='semibold', color='#2c6bb5')
    ax.text(105, i, f"-{drop:.0f} pp", va='center', ha='center', fontsize=10, fontweight='bold', color='#c0392b')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_station_degradation.png"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "fig3_station_degradation.pdf"))
if os.path.exists(PAPER_FIG_DIR):
    fig.savefig(os.path.join(PAPER_FIG_DIR, "fig3_station_degradation.png"), dpi=300)
    fig.savefig(os.path.join(PAPER_FIG_DIR, "fig3_station_degradation.pdf"))
plt.close(fig)
print("Saved fig3_station_degradation")
