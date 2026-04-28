#!/usr/bin/env python3
"""
VDR Ligand Lipinski/Veber Visualization
Produces a multi-panel figure saved as vdr_lipinski_plots.png
Panels:
  1. Histograms: MW, LogP, TPSA, HBD, HBA, RotBonds (by species)
  2. Bar chart: Lipinski pass/fail by species
  3. Scatter: MW vs LogP colored by species, shape by Lipinski pass/fail
  4. Radar: average descriptors per species
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np
except ImportError:
    raise SystemExit("pip install matplotlib numpy")

INPUT  = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_filtered.csv")
OUTPUT = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_lipinski_plots.png")

# ── Load data ─────────────────────────────────────────────────────────────────
with open(INPUT, newline="", encoding="utf-8") as fh:
    sample = fh.read(2000); fh.seek(0)
    delim = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(fh, delimiter=delim)
    all_rows = list(reader)

# Keep only rows with valid numeric descriptors
rows = [r for r in all_rows if r.get("MW","").strip() and
        r.get("Lipinski_pass","").strip() not in ("FAILED_PARSE","boron_cluster_excluded","")]

print(f"Total rows     : {len(all_rows)}")
print(f"Valid for plots: {len(rows)}")
print(f"Excluded       : {len(all_rows)-len(rows)} (boron clusters / parse failures)")

SPECIES   = ["Human", "Rat", "Zebrafish"]
COLORS    = {"Human": "#2196F3", "Rat": "#FF5722", "Zebrafish": "#4CAF50"}
MARKERS   = {"PASS": "o", "FAIL": "X"}
ALPHA     = 0.72

def vals(rows, col, species=None):
    subset = [r for r in rows if r["Species"] == species] if species else rows
    return [float(r[col]) for r in subset if r.get(col,"").strip()]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#FAFAFA")

# Title
fig.suptitle("VDR Ligand Physicochemical Properties\nLipinski / Veber Analysis by Species",
             fontsize=16, fontweight="bold", y=0.98)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1: Six histograms (2 rows × 3 cols), top portion
# ─────────────────────────────────────────────────────────────────────────────
DESCRIPTORS = [
    ("MW",       "Molecular Weight (Da)",        500,  None),
    ("LogP",     "LogP",                         5,    None),
    ("TPSA",     "TPSA (Å²)",                    140,  None),
    ("HBD",      "H-Bond Donors",                5,    None),
    ("HBA",      "H-Bond Acceptors",             10,   None),
    ("RotBonds", "Rotatable Bonds",              10,   None),
]

hist_axes = []
for i, (col, label, threshold, _) in enumerate(DESCRIPTORS):
    ax = fig.add_subplot(4, 3, i + 1)
    hist_axes.append(ax)
    ax.set_facecolor("#F5F5F5")

    all_vals = vals(rows, col)
    bin_min  = min(all_vals) if all_vals else 0
    bin_max  = max(all_vals) if all_vals else 1
    bins     = np.linspace(bin_min, bin_max, 20)

    for sp in SPECIES:
        v = vals(rows, col, sp)
        if v:
            ax.hist(v, bins=bins, alpha=0.65, color=COLORS[sp],
                    label=sp, edgecolor="white", linewidth=0.5)

    # Threshold line
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold})")

    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2: Stacked bar — Lipinski pass/fail by species  (row 3, col 1-2)
# ─────────────────────────────────────────────────────────────────────────────
ax_bar = fig.add_subplot(4, 3, (7, 8))
ax_bar.set_facecolor("#F5F5F5")

bar_data = {}
for sp in SPECIES:
    sp_rows = [r for r in rows if r["Species"] == sp]
    bar_data[sp] = {
        "PASS": sum(1 for r in sp_rows if r["Lipinski_pass"] == "PASS"),
        "FAIL": sum(1 for r in sp_rows if r["Lipinski_pass"] == "FAIL"),
    }

x      = np.arange(len(SPECIES))
width  = 0.35
passes = [bar_data[sp]["PASS"] for sp in SPECIES]
fails  = [bar_data[sp]["FAIL"] for sp in SPECIES]

bars1 = ax_bar.bar(x, passes, width, label="PASS", color="#4CAF50", edgecolor="white")
bars2 = ax_bar.bar(x, fails,  width, bottom=passes, label="FAIL",
                   color="#F44336", edgecolor="white")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(SPECIES, fontsize=10)
ax_bar.set_ylabel("Number of ligands", fontsize=10)
ax_bar.set_title("Lipinski Rule of Five — Pass / Fail by Species", fontsize=11, fontweight="bold")
ax_bar.legend(fontsize=9)
ax_bar.spines[["top","right"]].set_visible(False)
ax_bar.grid(axis="y", alpha=0.3)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax_bar.text(bar.get_x() + bar.get_width()/2, h/2,
                    str(int(h)), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
for bar, bot in zip(bars2, passes):
    h = bar.get_height()
    if h > 0:
        ax_bar.text(bar.get_x() + bar.get_width()/2, bot + h/2,
                    str(int(h)), ha="center", va="center", fontsize=9, color="white", fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3: MW vs LogP scatter  (row 3, col 3)
# ─────────────────────────────────────────────────────────────────────────────
ax_sc = fig.add_subplot(4, 3, 9)
ax_sc.set_facecolor("#F5F5F5")

for sp in SPECIES:
    for lip_status, marker in MARKERS.items():
        subset = [r for r in rows if r["Species"] == sp and r["Lipinski_pass"] == lip_status]
        if subset:
            x_vals = [float(r["MW"])   for r in subset]
            y_vals = [float(r["LogP"]) for r in subset]
            ax_sc.scatter(x_vals, y_vals, c=COLORS[sp], marker=marker,
                          alpha=ALPHA, s=55, edgecolors="white", linewidths=0.4)

# Threshold lines
ax_sc.axvline(500, color="red",  linestyle="--", linewidth=1.2, alpha=0.7)
ax_sc.axhline(5,   color="blue", linestyle="--", linewidth=1.2, alpha=0.7)
ax_sc.text(505, ax_sc.get_ylim()[0] + 0.2, "MW=500", color="red",  fontsize=7)
ax_sc.text(ax_sc.get_xlim()[0]+5, 5.1,    "LogP=5", color="blue", fontsize=7)

ax_sc.set_xlabel("Molecular Weight (Da)", fontsize=9)
ax_sc.set_ylabel("LogP",                  fontsize=9)
ax_sc.set_title("MW vs LogP by Species",  fontsize=10, fontweight="bold")
ax_sc.spines[["top","right"]].set_visible(False)
ax_sc.grid(alpha=0.3)

legend_elements = (
    [mpatches.Patch(color=COLORS[sp], label=sp) for sp in SPECIES] +
    [Line2D([0],[0], marker="o", color="gray", label="Lipinski PASS", linestyle="None", markersize=7),
     Line2D([0],[0], marker="X", color="gray", label="Lipinski FAIL", linestyle="None", markersize=7)]
)
ax_sc.legend(handles=legend_elements, fontsize=7, loc="upper left")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4: Radar chart — average descriptors per species  (row 4, cols 1-3)
# ─────────────────────────────────────────────────────────────────────────────
ax_radar = fig.add_subplot(4, 3, (10, 12), polar=True)

radar_cols    = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]
radar_labels  = ["MW\n(/500)", "LogP\n(/5)", "TPSA\n(/140)", "HBD\n(/5)", "HBA\n(/10)", "RotBonds\n(/10)"]
thresholds    = [500, 5, 140, 5, 10, 10]
N             = len(radar_cols)
angles        = [n / float(N) * 2 * math.pi for n in range(N)]
angles       += angles[:1]

ax_radar.set_theta_offset(math.pi / 2)
ax_radar.set_theta_direction(-1)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_labels, fontsize=9)
ax_radar.set_ylim(0, 1.2)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(["25%","50%","75%","100%"], fontsize=7, color="grey")
ax_radar.grid(color="grey", alpha=0.3)

for sp in SPECIES:
    sp_rows = [r for r in rows if r["Species"] == sp]
    if not sp_rows:
        continue
    norm_vals = []
    for col, thresh in zip(radar_cols, thresholds):
        v = [float(r[col]) for r in sp_rows if r.get(col,"").strip()]
        avg = sum(v)/len(v) if v else 0
        norm_vals.append(min(avg / thresh, 1.5))   # cap at 1.5x threshold
    norm_vals += norm_vals[:1]
    ax_radar.plot(angles, norm_vals, color=COLORS[sp], linewidth=2, label=sp)
    ax_radar.fill(angles, norm_vals, color=COLORS[sp], alpha=0.15)

# Threshold circle at 1.0
ax_radar.plot(angles, [1.0]*len(angles), color="red", linestyle="--",
              linewidth=1.2, alpha=0.6, label="Threshold")

ax_radar.set_title("Average Descriptor Profile by Species\n(normalised to Lipinski/Veber threshold)",
                   fontsize=10, fontweight="bold", pad=20)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nFigure saved → {OUTPUT}")
