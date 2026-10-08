import os
import json
import numpy as np
import matplotlib.pyplot as plt

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

with open(os.path.join(RESULTS_DIR, "retraining_experiment.json"), "r") as f:
    res = json.load(f)

runs = res["runs"]
contrasts = res["contrasts"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# ---------------- Panel A: Catalog Bursts Recall Gap across Seeds ----------------
# Averages and std across seeds
pub_leaked = runs['published_FlareSense-v2']['recall_leaked_catalog'] * 100
pub_clean = runs['published_FlareSense-v2']['recall_clean_catalog'] * 100

rc_leaked = [runs[f'random_control_seed{s}']['recall_leaked_catalog'] * 100 for s in range(3)]
rc_clean = [runs[f'random_control_seed{s}']['recall_clean_catalog'] * 100 for s in range(3)]

p_leaked = [runs[f'purged_seed{s}']['recall_leaked_catalog'] * 100 for s in range(3)]
p_clean = [runs[f'purged_seed{s}']['recall_clean_catalog'] * 100 for s in range(3)]

models = ['Published v2\n(Uncontrolled)', 'Random Control\n(3-Seed Mean ± Std)', 'Purged\n(3-Seed Mean ± Std)']
leaked_means = [pub_leaked, np.mean(rc_leaked), np.mean(p_leaked)]
leaked_errs = [0, np.std(rc_leaked, ddof=1), np.std(p_leaked, ddof=1)]

clean_means = [pub_clean, np.mean(rc_clean), np.mean(p_clean)]
clean_errs = [0, np.std(rc_clean, ddof=1), np.std(p_clean, ddof=1)]

x = np.arange(len(models))
width = 0.32

rects1 = ax1.bar(x - width/2, leaked_means, width, yerr=leaked_errs, capsize=4,
                 label='Leaked Bursts (Event in Train)', color='#e05252', edgecolor='#b33939', alpha=0.9)
rects2 = ax1.bar(x + width/2, clean_means, width, yerr=clean_errs, capsize=4,
                 label='Clean Bursts (Event Unseen)', color='#4a90e2', edgecolor='#2c6bb5', alpha=0.9)

ax1.set_ylabel('Recall on Catalog Bursts (%)', fontsize=11, fontweight='bold')
ax1.set_title('A: Controlled Retraining — Catalog-Harmonized Recall', fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10, fontweight='semibold')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.6)
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)

# Value annotations
for i in range(len(models)):
    l_val = leaked_means[i]
    c_val = clean_means[i]
    gap = l_val - c_val
    ax1.text(i - width/2, l_val + 2.5 + leaked_errs[i], f"{l_val:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#b33939')
    ax1.text(i + width/2, c_val + 2.5 + clean_errs[i], f"{c_val:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c6bb5')
    ax1.text(i, max(l_val, c_val) + 9, f"Gap:\n{gap:.1f} pp", ha='center', va='bottom', fontsize=9, fontweight='heavy', color='#444444')

# ---------------- Panel B: Forest Plot of Causal Double-Difference (DD) ----------------
# Catalog gap DD for Seed 0, Seed 1, Seed 2, and Pooled 3-Seed Mean
seeds = [0, 1, 2]
cat_estimates = [contrasts[f'random_control_seed{s}_minus_purged_seed{s}']['estimate_pp']['delta_gap_catalog'] for s in seeds]
cat_ci_low = [contrasts[f'random_control_seed{s}_minus_purged_seed{s}']['ci95_pp_cluster']['delta_gap_catalog'][0] for s in seeds]
cat_ci_high = [contrasts[f'random_control_seed{s}_minus_purged_seed{s}']['ci95_pp_cluster']['delta_gap_catalog'][1] for s in seeds]

mean_est = np.mean(cat_estimates)
std_est = np.std(cat_estimates, ddof=1)
# 95% t-interval for 3 seeds
ci_mean_low = mean_est - 4.303 * (std_est / np.sqrt(3))
ci_mean_high = mean_est + 4.303 * (std_est / np.sqrt(3))

items = [
    'Seed 0',
    'Seed 1',
    'Seed 2',
    'Pooled Mean (N=3)'
]
all_est = cat_estimates + [mean_est]
all_low = cat_ci_low + [ci_mean_low]
all_high = cat_ci_high + [ci_mean_high]

y_pos = np.arange(len(items))
colors = ['#8e44ad', '#8e44ad', '#8e44ad', '#27ae60']

xerr_left = [all_est[i] - all_low[i] for i in range(len(items))]
xerr_right = [all_high[i] - all_est[i] for i in range(len(items))]

ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, zorder=1, label='Null Hypothesis (ΔΔ = 0, No Leakage Effect)')

for i in range(len(items)):
    ax2.errorbar(all_est[i], y_pos[i], xerr=[[xerr_left[i]], [xerr_right[i]]], fmt='o',
                 color=colors[i], ecolor=colors[i], elinewidth=2.5, capsize=6, capthick=2, markersize=8, zorder=3)
    ax2.plot(all_est[i], y_pos[i], marker='D' if i == 3 else 's', markersize=9, color=colors[i], zorder=4)
    ax2.text(all_high[i] + 0.4, y_pos[i], f"+{all_est[i]:.2f} pp [95% CI: {all_low[i]:.1f}, {all_high[i]:.1f}]",
             va='center', ha='left', fontsize=9.5, fontweight='bold', color=colors[i])

ax2.set_yticks(y_pos)
ax2.set_yticklabels(items, fontsize=11, fontweight='bold')
ax2.set_xlabel('Causal Catalog Gap Reduction ΔΔ (Percentage Points, 95% CI)', fontsize=11, fontweight='bold')
ax2.set_title('B: Causal Double-Difference Across Independent Seeds', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlim(-1, 19.0)
ax2.grid(axis='x', alpha=0.6)
ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_causal_retraining_contrast.png"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "fig4_causal_retraining_contrast.pdf"))
plt.close(fig)
print("Updated fig4_causal_retraining_contrast with complete 3-seed multi-run aggregation (PNG and PDF)")
