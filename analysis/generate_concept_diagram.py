import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIG_DIR = r"C:\Users\User\Desktop\FlareSense-v2-Audit\figures"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), dpi=300)

# ----------------- Subplot 1: Random Split (FLAWED) -----------------
ax1.set_title("A: Standard Random Split (FlareSense-v2, arXiv:2607.26014) — Multi-Station Leakage", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 40)
ax1.axis('off')

# Timeline axis
ax1.annotate("", xy=(95, 20), xytext=(5, 20), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
ax1.text(96, 20, "Time (t)", va='center', fontsize=11, fontweight='bold', color="#333333")

# Solar Burst Event 1
burst1 = patches.Rectangle((20, 5), 25, 30, linewidth=1.5, edgecolor='#e74c3c', facecolor='#fadbd8', linestyle='--', alpha=0.5)
ax1.add_patch(burst1)
ax1.text(32.5, 36.5, "Solar Burst Event #1 (12:00 - 12:15 UTC)", ha='center', fontsize=10, fontweight='bold', color='#c0392b')

# Antenna recordings in Event 1
stations = ["Station A (ALMATY)", "Station B (KASI)", "Station C (MEXART)", "Station D (MRO)"]
y_offsets = [28, 22, 16, 10]
splits_random = ["TRAIN", "TRAIN", "TEST (Leaked!)", "VAL"]
colors_random = ["#27ae60", "#27ae60", "#e74c3c", "#f39c12"]

for s, y, sp, c in zip(stations, y_offsets, splits_random, colors_random):
    rect = patches.Rectangle((22, y - 2), 21, 4, facecolor=c, edgecolor='#333333', lw=1, alpha=0.9, zorder=3)
    ax1.add_patch(rect)
    ax1.text(18, y, s, ha='right', va='center', fontsize=9, fontweight='bold')
    ax1.text(32.5, y, sp, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrow pointing to leakage
ax1.annotate("Model memorizes event physics\nfrom Stations A & B in TRAIN,\nartificially inflating TEST score!",
             xy=(43, 16), xytext=(55, 16),
             va='center', fontsize=10, fontweight='bold', color='#c0392b',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdedec', edgecolor='#c0392b', lw=1.5),
             arrowprops=dict(arrowstyle='->', lw=2, color='#c0392b'))


# ----------------- Subplot 2: Event-Grouped Split (PROPOSED AUDIT SOLUTION) -----------------
ax2.set_title("B: Proposed Leak-Free Event-Grouped Split (Ours) — Coordinated Sensor Blocking", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 40)
ax2.axis('off')

# Timeline axis
ax2.annotate("", xy=(95, 20), xytext=(5, 20), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
ax2.text(96, 20, "Time (t)", va='center', fontsize=11, fontweight='bold', color="#333333")

# Solar Burst Event 1
burst2 = patches.Rectangle((20, 5), 25, 30, linewidth=1.5, edgecolor='#2980b9', facecolor='#ebf5fb', linestyle='--', alpha=0.5)
ax2.add_patch(burst2)
ax2.text(32.5, 36.5, "Solar Burst Event #1 (Entire Window Assigned to TEST)", ha='center', fontsize=10, fontweight='bold', color='#2471a3')

splits_clean = ["TEST", "TEST", "TEST", "TEST"]
colors_clean = ["#2980b9", "#2980b9", "#2980b9", "#2980b9"]

for s, y, sp, c in zip(stations, y_offsets, splits_clean, colors_clean):
    rect = patches.Rectangle((22, y - 2), 21, 4, facecolor=c, edgecolor='#333333', lw=1, alpha=0.9, zorder=3)
    ax2.add_patch(rect)
    ax2.text(18, y, s, ha='right', va='center', fontsize=9, fontweight='bold')
    ax2.text(32.5, y, sp, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrow pointing to clean isolation
ax2.annotate("Strict Event-Group Isolation:\nAll parallel station observations\nof the same physical burst stay in the same split.\nTest positive event leakage = 0.0%",
             xy=(43, 20), xytext=(55, 20),
             va='center', fontsize=10, fontweight='bold', color='#2471a3',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#eaf2f8', edgecolor='#2471a3', lw=1.5),
             arrowprops=dict(arrowstyle='->', lw=2, color='#2471a3'))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig0_concept_leakage_mechanism.png"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "fig0_concept_leakage_mechanism.pdf"))
plt.close(fig)
print("Saved fig0_concept_leakage_mechanism")
