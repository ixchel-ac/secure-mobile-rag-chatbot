"""Generate leaderboard analysis charts: heatmap + bar chart."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

# ── Data (from leaderboard_export_2026-05-12.csv) ──────────────────────

profiles = [
    "fw_l1_hardened_fw_l2_bert",
    "fw_l1_hardened",
    "fw_l1_hardened_fw_l2_base",
    "fw_l1_naive_fw_l2_bert",
    "fw_l1_naive",
    "hardened_fw_l2_bert",
    "fw_l1_naive_fw_l2_base",
    "hardened",
    "hardened_fw_l2_base",
    "naive_fw_l2_bert",
    "naive",
    "naive_fw_l2_base",
]

# Short labels for display
labels = [
    "FW-L1 + Hardened + FW-L2-BERT",
    "FW-L1 + Hardened",
    "FW-L1 + Hardened + FW-L2-spaCy",
    "FW-L1 + Naive + FW-L2-BERT",
    "FW-L1 + Naive",
    "Hardened + FW-L2-BERT",
    "FW-L1 + Naive + FW-L2-spaCy",
    "Hardened (BASELINE)",
    "Hardened + FW-L2-spaCy",
    "Naive + FW-L2-BERT",
    "Naive",
    "Naive + FW-L2-spaCy",
]

compound  = [91.75, 91.68, 89.36, 85.32, 82.65, 81.42, 81.10, 81.02, 78.80, 68.45, 63.20, 61.11]
accuracy  = [73.90, 73.76, 72.29, 80.71, 81.33, 68.51, 77.81, 68.08, 67.94, 51.48, 52.14, 50.64]
precision = [71.41, 71.27, 69.98, 86.64, 86.95, 68.74, 84.84, 68.49, 68.16, 96.49, 96.13, 92.63]
recall    = [97.48, 97.41, 92.22, 66.30, 68.07, 91.04, 56.81, 90.97, 88.67,  8.15,  9.19,  4.67]
f1        = [82.45, 82.33, 79.57, 75.10, 76.33, 78.33, 68.05, 78.15, 77.08, 15.03, 16.78,  8.89]
latency   = [ 3.28,  2.98,  3.02,  3.76,  3.56,  3.26,  3.72,  3.29,  3.31,  4.24,  4.31,  4.08]


# ── Chart 1: Heatmap ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 8))

metrics = ["Compound", "Accuracy", "Precision", "Recall", "F1"]
data = np.array([compound, accuracy, precision, recall, f1]).T  # shape: (12, 5)

im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

# Axis labels — place metric names on top to avoid tick overlap
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True, pad=8)
ax.set_yticks(range(len(labels)))
ranked_labels = [f"#{i+1}  {l}" for i, l in enumerate(labels)]
ax.set_yticklabels(ranked_labels, fontsize=11, ha="right")
ax.tick_params(axis="y", pad=15)

# Annotate cells with values
for i in range(len(labels)):
    for j in range(len(metrics)):
        val = data[i, j]
        color = "white" if val < 40 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Score (%)", fontsize=11)

ax.set_title("RAG Firewall Profile Comparison — Classification Metrics",
             fontsize=14, fontweight="bold", pad=15)

plt.subplots_adjust(left=0.25)
plt.savefig("leaderboard_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("[charts] Saved leaderboard_heatmap.png")


# ── Chart 2: Grouped Bar Chart (Compound + F1 + Latency) ──────────────

fig, ax1 = plt.subplots(figsize=(14, 6))

x = np.arange(len(labels))
width = 0.30

bars1 = ax1.bar(x - width, compound, width, label="Compound Score",
                color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.5)
bars2 = ax1.bar(x, f1, width, label="F1 Score",
                color="#4CAF50", alpha=0.85, edgecolor="white", linewidth=0.5)

ax1.set_ylabel("Score (0–100)", fontsize=12, fontweight="bold")
ax1.set_ylim(0, 105)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1565C0")
for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2E7D32")

# Latency on secondary axis
ax2 = ax1.twinx()
bars3 = ax2.bar(x + width, latency, width, label="Avg Latency (s)",
                color="#FF9800", alpha=0.85, edgecolor="white", linewidth=0.5)
ax2.set_ylabel("Latency (seconds)", fontsize=12, fontweight="bold", color="#E65100")
ax2.set_ylim(0, 6)
ax2.tick_params(axis="y", labelcolor="#E65100")

for bar in bars3:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.1f}s",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#E65100")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10,
           framealpha=0.9)

ax1.set_title("Compound Score, F1, and Latency by Profile",
              fontsize=14, fontweight="bold", pad=15)
ax1.axhline(y=90, color="#1565C0", linestyle="--", alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig("leaderboard_bars.png", dpi=200, bbox_inches="tight")
plt.close()
print("[charts] Saved leaderboard_bars.png")


# ── Chart 3: Precision vs Recall Scatter ───────────────────────────────

fig, ax = plt.subplots(figsize=(11, 7))

# Color by prompt family
def get_color(p):
    if "fw_l1" in p and "hardened" in p and "naive" not in p:
        return "#2196F3"   # FW-L1 + hardened: blue
    elif "fw_l1" in p and "naive" in p:
        return "#FF9800"   # FW-L1 + naive: orange
    elif "hardened" in p and "naive" not in p:
        return "#4CAF50"   # hardened only: green
    else:
        return "#F44336"   # naive only: red

# Shape by FW-L2 variant: triangle=none, circle=BERT, square=spaCy
def get_marker(p):
    if "bert" in p:
        return "o"   # circle for BERT
    elif "base" in p:
        return "s"   # square for spaCy
    else:
        return "^"   # triangle for no FW-L2

# Scale sizes so differences are visible: shift baseline to 0, then amplify
min_compound = min(compound)
sizes = [((s - min_compound + 5) ** 2) * 0.8 for s in compound]

# Plot each point individually to support different markers
for i, p in enumerate(profiles):
    ax.scatter(recall[i], precision[i], s=sizes[i], c=get_color(p),
               marker=get_marker(p), alpha=0.80, edgecolors="white",
               linewidth=1.5, zorder=5)

# Quadrant labels — positioned inside each quadrant
ax.axhline(y=80, color="gray", linestyle=":", alpha=0.4)
ax.axvline(x=80, color="gray", linestyle=":", alpha=0.4)
ax.text(90, 58, "High Recall\nLow Precision\n(over-blocks)", fontsize=8,
        ha="center", color="gray", style="italic")
ax.text(30, 96, "Low Recall\nHigh Precision\n(under-blocks)", fontsize=8,
        ha="center", color="gray", style="italic")
ax.text(90, 93, "Ideal\nZone", fontsize=9, ha="center", color="green",
        fontweight="bold", alpha=0.5)
ax.text(55, 72, "Low Recall\nLow Precision", fontsize=8,
        ha="center", va="center", color="gray", style="italic")

ax.set_xlabel("Recall (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Precision (%)", fontsize=12, fontweight="bold")
ax.set_xlim(-2, 105)
ax.set_ylim(55, 100)
ax.set_title("Comparative View of Precision, Recall and Compound Scores",
             fontsize=14, fontweight="bold", pad=15)

# Legend: 3 columns — color (prompt family) | shape (FW-L2) | bubble size
# matplotlib ncol fills column-major, so group items sequentially per column
from matplotlib.lines import Line2D
_blank = Line2D([0], [0], color="w", marker="None", label="")
legend_elements = [
    # Column 1: prompt family (color)
    Line2D([0], [0], color="#2196F3", linewidth=4, label="FW-L1 + Hardened"),
    Line2D([0], [0], color="#FF9800", linewidth=4, label="FW-L1 + Naive"),
    Line2D([0], [0], color="#4CAF50", linewidth=4, label="Hardened (no FW-L1)"),
    Line2D([0], [0], color="#F44336", linewidth=4, label="Naive (no FW-L1)"),
    # Column 2: FW-L2 variant (shape)
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
           markersize=10, label="No FW-L2"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markersize=10, label="FW-L2-BERT"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
           markersize=10, label="FW-L2-spaCy"),
    _blank,
    # Column 3: bubble size
    Line2D([0], [0], color="w", marker="None", label="Bubble size ="),
    Line2D([0], [0], color="w", marker="None", label="Compound Score"),
    Line2D([0], [0], color="w", marker="None", label="(larger = higher)"),
    _blank,
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9,
          ncol=3, columnspacing=1.5)

plt.tight_layout()
plt.savefig("leaderboard_precision_recall.png", dpi=200, bbox_inches="tight")
plt.close()
print("[charts] Saved leaderboard_precision_recall.png")
