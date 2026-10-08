"""generate_comprehensive_figure.py

Generates the definitive 2-panel publication figure:
  - Panel A: Recall on Catalog Bursts Across Operating Points (Published v2 vs Calib-Fixed vs Matched FPR)
  - Panel B: Causal Contrast Forest Plot (Double-Difference ΔΔ across regimes, N=3 seeds)

Outputs:
  figures/fig4_causal_retraining_contrast.png
  figures/fig4_causal_retraining_contrast.pdf
"""

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

with open(os.path.join(RESULTS_DIR, "decomposition_and_robustness.json"), "r") as f:
    res = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300)
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.16, wspace=0.32)

# -----------------------------------------------------------------------------
# Panel A: Operating Regime Sensitivity (Catalog Bursts)
# -----------------------------------------------------------------------------
models = [
    'Published v2\n(Uncontrolled)',
    'Calib-Fixed (Purged)\n(Test FPR = 0.15%)',
    'Matched FPR (Purged)*\n(Test FPR = 0.84%)'
]

pub_lc = res['regime_calib_fixed']['published_FlareSense-v2']['recall_leaked_catalog'] * 100
pub_cc = res['regime_calib_fixed']['published_FlareSense-v2']['recall_clean_catalog'] * 100

cf_lc = np.mean([res['regime_calib_fixed'][f'purged_seed{s}']['recall_leaked_catalog'] * 100 for s in range(3)])
cf_lc_err = np.std([res['regime_calib_fixed'][f'purged_seed{s}']['recall_leaked_catalog'] * 100 for s in range(3)], ddof=1)
cf_cc = np.mean([res['regime_calib_fixed'][f'purged_seed{s}']['recall_clean_catalog'] * 100 for s in range(3)])
cf_cc_err = np.std([res['regime_calib_fixed'][f'purged_seed{s}']['recall_clean_catalog'] * 100 for s in range(3)], ddof=1)

mfpr_lc = np.mean([res['regime_test_matched_fpr'][f'purged_seed{s}']['recall_leaked_catalog'] * 100 for s in range(3)])
mfpr_lc_err = np.std([res['regime_test_matched_fpr'][f'purged_seed{s}']['recall_leaked_catalog'] * 100 for s in range(3)], ddof=1)
mfpr_cc = np.mean([res['regime_test_matched_fpr'][f'purged_seed{s}']['recall_clean_catalog'] * 100 for s in range(3)])
mfpr_cc_err = np.std([res['regime_test_matched_fpr'][f'purged_seed{s}']['recall_clean_catalog'] * 100 for s in range(3)], ddof=1)

x = np.arange(len(models))
width = 0.32

l_means = [pub_lc, cf_lc, mfpr_lc]
l_errs = [0, cf_lc_err, mfpr_lc_err]
c_means = [pub_cc, cf_cc, mfpr_cc]
c_errs = [0, cf_cc_err, mfpr_cc_err]

ax1.bar(x - width/2, l_means, width, yerr=l_errs, capsize=4, label='Leaked Bursts (Event in Train)',
        color='#e05252', edgecolor='#b33939', alpha=0.9)
ax1.bar(x + width/2, c_means, width, yerr=c_errs, capsize=4, label='Clean Bursts (Event Unseen)',
        color='#4a90e2', edgecolor='#2c6bb5', alpha=0.9)

ax1.set_ylabel('Recall on Catalog Bursts (%)', fontsize=11, fontweight='bold')
ax1.set_title('A: Recall on Catalog Bursts Across Operating Regimes', fontsize=11.5, fontweight='bold', pad=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9.0, fontweight='bold')
ax1.set_ylim(0, 115)
ax1.grid(axis='y', alpha=0.6)
ax1.legend(loc='upper center', bbox_to_anchor=(0.50, 0.98), ncol=1, frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc', fontsize=8.5)
ax1.text(0.02, -0.15, '*Matched FPR is an exploratory sensitivity check with threshold set on test split',
         transform=ax1.transAxes, fontsize=8.0, fontstyle='italic', color='#555555')

for i in range(len(models)):
    l_v = l_means[i]
    c_v = c_means[i]
    gap = l_v - c_v
    ax1.text(i - width/2, l_v + 1.8 + l_errs[i], f'{l_v:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#b33939')
    ax1.text(i + width/2, c_v + 1.8 + c_errs[i], f'{c_v:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2c6bb5')
    ax1.text(i, max(l_v, c_v) + 7.5, f'Gap:\n{gap:.1f} pp', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#333333')

items_b = [
    'Calib-Fixed (Catalog)',
    'Calib-Fixed (All Bursts)',
    'Matched Recall* (All Bursts)',
    'Matched FPR* (All Bursts)',
    'Matched FPR* (Catalog)'
]

cf_cat = res['contrasts']['calib_fixed']['pooled_3seed']['delta_gap_catalog']
cf_all = res['contrasts']['calib_fixed']['pooled_3seed']['delta_gap']
mr_all = res['contrasts']['test_matched_recall']['pooled_3seed']['delta_gap']
mf_all = res['contrasts']['test_matched_fpr']['pooled_3seed']['delta_gap']
mf_cat = res['contrasts']['test_matched_fpr']['pooled_3seed']['delta_gap_catalog']

b_means = [cf_cat['mean'], cf_all['mean'], mr_all['mean'], mf_all['mean'], mf_cat['mean']]
b_lows = [cf_cat['ci95_t'][0], cf_all['ci95_t'][0], mr_all['ci95_t'][0], mf_all['ci95_t'][0], mf_cat['ci95_t'][0]]
b_highs = [cf_cat['ci95_t'][1], cf_all['ci95_t'][1], mr_all['ci95_t'][1], mf_all['ci95_t'][1], mf_cat['ci95_t'][1]]

y_pos_b = np.arange(len(items_b))
colors_b = ['#27ae60', '#2980b9', '#8e44ad', '#d35400', '#7f8c8d']

ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, zorder=1, label='Null Hypothesis (ΔΔ = 0)')

for i in range(len(items_b)):
    m = b_means[i]
    lo = b_lows[i]
    hi = b_highs[i]
    ax2.errorbar(m, y_pos_b[i], xerr=[[m - lo], [hi - m]], fmt='o',
                 color=colors_b[i], ecolor=colors_b[i], elinewidth=2.5, capsize=5, capthick=2, markersize=8, zorder=3)
    ax2.plot(m, y_pos_b[i], marker='D', markersize=8, color=colors_b[i], zorder=4)
    ax2.text(max(hi, m) + 0.4, y_pos_b[i], f'{m:+.2f} pp [{lo:.1f}, {hi:.1f}]',
             va='center', ha='left', fontsize=8.5, fontweight='bold', color=colors_b[i])

ax2.set_yticks(y_pos_b)
ax2.set_yticklabels(items_b, fontsize=9.0, fontweight='bold')
ax2.set_xlabel('Causal Gap Reduction ΔΔ (Percentage Points, 95% t-interval)', fontsize=10.5, fontweight='bold')
ax2.set_title('B: Causal Contrast Stability Across Regimes (N=3 Seeds)', fontsize=11.5, fontweight='bold', pad=14)
ax2.set_xlim(-6.5, 13.5)
ax2.grid(axis='x', alpha=0.6)
ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)
ax2.text(0.02, -0.15, '*Sensitivity analysis (threshold set on test); error bars: 95% t-interval over N=3 seeds',
         transform=ax2.transAxes, fontsize=8.0, fontstyle='italic', color='#555555')

PAPER_FIG_DIR = r"C:\Users\User\Desktop\FlareSense-v2-Audit\paper\figures"
fig.savefig(os.path.join(FIG_DIR, "fig4_causal_retraining_contrast.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, "fig4_causal_retraining_contrast.pdf"), bbox_inches='tight')
if os.path.exists(PAPER_FIG_DIR):
    fig.savefig(os.path.join(PAPER_FIG_DIR, "fig4_causal_retraining_contrast.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PAPER_FIG_DIR, "fig4_causal_retraining_contrast.pdf"), bbox_inches='tight')
plt.close(fig)
print("Saved refined comprehensive 2-panel figure: fig4_causal_retraining_contrast (PNG & PDF)")
