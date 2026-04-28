#!/usr/bin/env python3
"""
VDR Ligand Master Visualization Pipeline
==========================================
Input:  vdr_ligands_final.csv
Output: All figures for thesis discussion, in order:

  Fig 1 — Ligand class distribution bar chart
  Fig 2 — Lipinski/Veber descriptor histograms (by class)
  Fig 3 — Lipinski pass/fail by class bar chart
  Fig 4 — Tanimoto similarity heatmap ECFP4
  Fig 5 — Tanimoto similarity heatmap ECFP6
  Fig 6 — UMAP by ligand class ECFP6
  Fig 7 — UMAP by species ECFP6
  Fig 8 — UMAP by HDBSCAN cluster ECFP6
  Fig 9 — UMAP labeled by ligand class ECFP6
  Fig 10 — UMAP by ligand class ECFP4      (supplementary)
  Fig 11 — UMAP by HDBSCAN cluster ECFP4  (supplementary)

All saved to same folder as input CSV.
"""

import csv, warnings, math
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

# ── Dependencies check ────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("ERROR: conda install -c conda-forge rdkit")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
except ImportError:
    raise SystemExit("ERROR: pip install matplotlib")

try:
    import umap as umap_lib
except ImportError:
    raise SystemExit("ERROR: pip install umap-learn")

try:
    import hdbscan
except ImportError:
    raise SystemExit("ERROR: pip install hdbscan")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_final.csv")
OUTDIR = INPUT.parent

# ── Color palettes ────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "secosteroid":   "#1565C0",
    "non_steroidal": "#E65100",
    "steroidal":     "#2E7D32",
    "gemini":        "#6A1B9A",
    "boron_cluster": "#9E9E9E",
}
SPECIES_COLORS = {
    "Human":     "#2196F3",
    "Rat":       "#FF5722",
    "Zebrafish": "#4CAF50",
}
BORON_CODES = {"A1MAV","A1MAW","A1MAX","A1MAY","A1MAZ","A1MAU","M7E"}

# ── Load data ─────────────────────────────────────────────────────────────────
with open(INPUT, newline="", encoding="utf-8") as fh:
    sample = fh.read(2000); fh.seek(0)
    delim = ";" if sample.count(";") > sample.count(",") else ","
    rows = list(csv.DictReader(fh, delimiter=delim))

print(f"Loaded {len(rows)} rows from {INPUT.name}")

# Parse molecules
records = []
for row in rows:
    code   = row["PyMOL_Ligand_Name"].strip()
    smiles = row.get("SMILES_stereo","").strip() or row.get("SMILES","").strip()
    mol    = None
    if code not in BORON_CODES and smiles:
        mol = Chem.MolFromSmiles(smiles)
    records.append({
        "row":       row,
        "code":      code,
        "species":   row["Species"].strip(),
        "cls":       row["Ligand_Class"].strip(),
        "mol":       mol,
        "smiles":    smiles,
        "lip_pass":  row.get("Lipinski_pass","").strip(),
    })

valid  = [r for r in records if r["mol"] is not None]
boron  = [r for r in records if r["mol"] is None]
print(f"Valid for fingerprinting: {len(valid)}  |  Boron/excluded: {len(boron)}")

# ── Recompute Lipinski/Veber descriptors ──────────────────────────────────────
def get_props(mol):
    if mol is None:
        return None
    return {
        "MW":       round(Descriptors.ExactMolWt(mol), 2),
        "HBD":      rdMolDescriptors.CalcNumHBD(mol),
        "HBA":      rdMolDescriptors.CalcNumHBA(mol),
        "LogP":     round(Descriptors.MolLogP(mol), 2),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "TPSA":     round(rdMolDescriptors.CalcTPSA(mol), 2),
    }

for rec in valid:
    rec["props"] = get_props(rec["mol"])

# ── Compute fingerprints + UMAP + HDBSCAN ────────────────────────────────────
print("\nComputing fingerprints ...")
fp_data = {}
for radius in [2, 3]:
    ecfp = f"ECFP{radius*2}"
    fps  = [AllChem.GetMorganFingerprintAsBitVect(r["mol"], radius, nBits=2048) for r in valid]

    # Matrix
    arr = np.zeros((len(fps), 2048), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])

    # Tanimoto
    n   = len(fps)
    tan = np.zeros((n, n))
    for i in range(n):
        tan[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)

    # UMAP
    reducer   = umap_lib.UMAP(n_components=2, n_neighbors=15,
                               min_dist=0.1, metric="jaccard", random_state=42)
    embedding = reducer.fit_transform(arr)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric="euclidean")
    labels    = clusterer.fit_predict(embedding)
    n_cl      = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  {ecfp}: {n_cl} clusters, {sum(1 for l in labels if l==-1)} noise points")

    fp_data[ecfp] = dict(fps=fps, arr=arr, tan=tan, emb=embedding, labels=labels, n_cl=n_cl)

# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Ligand class distribution bar chart
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Ligand class distribution ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#FAFAFA")
fig.suptitle("VDR Ligand Dataset — Class Distribution\nn=160 unique ligands",
             fontsize=14, fontweight="bold")

# Overall counts
class_order = ["secosteroid","non_steroidal","steroidal","gemini","boron_cluster"]
class_labels = ["Secosteroid","Non-steroidal","Steroidal","Gemini","Boron cluster"]
counts = [sum(1 for r in rows if r["Ligand_Class"]==c) for c in class_order]
colors = [CLASS_COLORS[c] for c in class_order]

ax = axes[0]
ax.set_facecolor("#F5F5F5")
bars = ax.bar(class_labels, counts, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
for bar, n in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            str(n), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of ligands", fontsize=11)
ax.set_title("Overall class distribution", fontsize=11, fontweight="bold")
ax.set_xticklabels(class_labels, rotation=20, ha="right", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# By species stacked
ax2 = axes[1]
ax2.set_facecolor("#F5F5F5")
species_order = ["Human","Rat","Zebrafish"]
x = np.arange(len(class_order))
width = 0.25
for i, sp in enumerate(species_order):
    sp_counts = [sum(1 for r in rows if r["Ligand_Class"]==c and r["Species"]==sp)
                 for c in class_order]
    ax2.bar(x + i*width, sp_counts, width, label=sp,
            color=SPECIES_COLORS[sp], edgecolor="white", linewidth=0.5, alpha=0.9)
ax2.set_xticks(x + width)
ax2.set_xticklabels(class_labels, rotation=20, ha="right", fontsize=10)
ax2.set_ylabel("Number of ligands", fontsize=11)
ax2.set_title("Class distribution by species", fontsize=11, fontweight="bold")
ax2.legend(fontsize=10)
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
p = OUTDIR / "Fig1_class_distribution.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Lipinski/Veber descriptor histograms by class
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 2] Descriptor histograms ...")
DESCRIPTORS = [
    ("MW",       "Molecular Weight (Da)", 500),
    ("LogP",     "LogP",                  5),
    ("TPSA",     "TPSA (Å²)",             140),
    ("HBD",      "H-Bond Donors",         5),
    ("HBA",      "H-Bond Acceptors",      10),
    ("RotBonds", "Rotatable Bonds",       10),
]

plot_classes = ["secosteroid","non_steroidal","steroidal","gemini"]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor("#FAFAFA")
fig.suptitle("VDR Ligand Physicochemical Descriptors by Ligand Class\n(boron cluster excluded)",
             fontsize=13, fontweight="bold")

for ax, (col, label, thresh) in zip(axes.flat, DESCRIPTORS):
    ax.set_facecolor("#F5F5F5")
    all_vals = [r["props"][col] for r in valid if r["cls"] in plot_classes]
    bins = np.linspace(min(all_vals), max(all_vals), 22)
    for cls in plot_classes:
        vals = [r["props"][col] for r in valid if r["cls"]==cls]
        if vals:
            ax.hist(vals, bins=bins, alpha=0.65, color=CLASS_COLORS[cls],
                    label=cls.replace("_"," "), edgecolor="white", linewidth=0.4)
    ax.axvline(thresh, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({thresh})")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=7)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
p = OUTDIR / "Fig2_descriptor_histograms.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — Lipinski pass/fail by class
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 3] Lipinski pass/fail by class ...")
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#F5F5F5")

plot_cls  = ["secosteroid","non_steroidal","steroidal","gemini"]
plot_lbls = ["Secosteroid","Non-steroidal","Steroidal","Gemini"]
x = np.arange(len(plot_cls))
w = 0.28

pass_counts = [sum(1 for r in valid if r["cls"]==c and r["lip_pass"]=="PASS") for c in plot_cls]
fail_counts = [sum(1 for r in valid if r["cls"]==c and r["lip_pass"]=="FAIL") for c in plot_cls]
total       = [sum(1 for r in valid if r["cls"]==c) for c in plot_cls]

b1 = ax.bar(x-w/2, pass_counts, w, label="Lipinski PASS", color="#43A047", edgecolor="white")
b2 = ax.bar(x+w/2, fail_counts, w, label="Lipinski FAIL", color="#E53935", edgecolor="white")

for bar, n in zip(b1, pass_counts):
    if n: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                  str(n), ha="center", fontsize=10, fontweight="bold")
for bar, n in zip(b2, fail_counts):
    if n: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                  str(n), ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(plot_lbls, fontsize=11)
ax.set_ylabel("Number of ligands", fontsize=11)
ax.set_title("Lipinski Rule of Five — Pass / Fail by Ligand Class\n(boron cluster excluded)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
p = OUTDIR / "Fig3_lipinski_by_class.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGS 4 & 5 — Tanimoto heatmaps
# ═════════════════════════════════════════════════════════════════════════════
def save_heatmap(ecfp, fignum):
    print(f"[Fig {fignum}] Tanimoto heatmap {ecfp} ...")
    tan = fp_data[ecfp]["tan"]
    n   = len(valid)
    order  = sorted(range(n), key=lambda i: (valid[i]["cls"], valid[i]["species"]))
    mat_s  = tan[np.ix_(order, order)]
    classes= [valid[i]["cls"] for i in order]
    codes  = [valid[i]["code"] for i in order]
    row_colors = [CLASS_COLORS.get(c,"#999") for c in classes]

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor("#FAFAFA")
    im = ax.imshow(mat_s, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Tanimoto Similarity")

    for i, color in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-3, i-0.5), 2.2, 1, color=color, clip_on=False))

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Ligands (sorted by class)", fontsize=11)
    ax.set_ylabel("Ligands (sorted by class)", fontsize=11)
    ax.set_title(f"Tanimoto Similarity Heatmap — Morgan {ecfp}\n"
                 f"n={n} ligands, sorted by ligand class",
                 fontsize=13, fontweight="bold")

    legend = [mpatches.Patch(color=CLASS_COLORS[c], label=c.replace("_"," ").title())
              for c in class_order if any(r["cls"]==c for r in valid)]
    ax.legend(handles=legend, loc="upper right",
              bbox_to_anchor=(1.20, 1.0), fontsize=10, title="Class", title_fontsize=10)

    plt.tight_layout()
    p = OUTDIR / f"Fig{fignum}_heatmap_{ecfp.lower()}.png"
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {p.name}")

save_heatmap("ECFP4", 4)
save_heatmap("ECFP6", 5)

# ═════════════════════════════════════════════════════════════════════════════
# Helper: UMAP scatter
# ═════════════════════════════════════════════════════════════════════════════
def umap_scatter(ax, embedding, color_list, label_list, title, legend_handles):
    ax.set_facecolor("#F5F5F5")
    unique_labels = list(dict.fromkeys(label_list))
    for label in unique_labels:
        idx = [i for i,l in enumerate(label_list) if l==label]
        ax.scatter(embedding[idx,0], embedding[idx,1],
                   c=color_list[idx[0]] if isinstance(color_list, list) else color_list,
                   s=55, alpha=0.85, edgecolors="white", linewidths=0.4,
                   label=label, zorder=3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.2)
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, loc="best", framealpha=0.9)

# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — UMAP by ligand class ECFP6 (main figure)
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 6] UMAP by class ECFP6 ...")
emb = fp_data["ECFP6"]["emb"]
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#F5F5F5")

for cls in class_order:
    idx = [i for i,r in enumerate(valid) if r["cls"]==cls]
    if idx:
        ax.scatter(emb[idx,0], emb[idx,1], c=CLASS_COLORS[cls], s=60,
                   alpha=0.85, edgecolors="white", linewidths=0.4,
                   label=cls.replace("_"," ").title(), zorder=3)

legend = [mpatches.Patch(color=CLASS_COLORS[c],
          label=c.replace("_"," ").title()) for c in class_order
          if any(r["cls"]==c for r in valid)]
ax.legend(handles=legend, fontsize=11, loc="best", framealpha=0.9)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("VDR Ligand Chemical Space — Morgan ECFP6\nColored by Ligand Class  |  n=153",
             fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.grid(alpha=0.2)
plt.tight_layout()
p = OUTDIR / "Fig6_umap_ecfp6_class.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 — UMAP by species ECFP6
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 7] UMAP by species ECFP6 ...")
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#F5F5F5")

for sp in ["Human","Rat","Zebrafish"]:
    idx = [i for i,r in enumerate(valid) if r["species"]==sp]
    if idx:
        ax.scatter(emb[idx,0], emb[idx,1], c=SPECIES_COLORS[sp], s=60,
                   alpha=0.85, edgecolors="white", linewidths=0.4, label=sp, zorder=3)

ax.legend(fontsize=11, loc="best", framealpha=0.9)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("VDR Ligand Chemical Space — Morgan ECFP6\nColored by Species  |  n=153",
             fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.grid(alpha=0.2)
plt.tight_layout()
p = OUTDIR / "Fig7_umap_ecfp6_species.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 8 — UMAP by HDBSCAN cluster ECFP6
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 8] UMAP by cluster ECFP6 ...")
labels = fp_data["ECFP6"]["labels"]
n_cl   = fp_data["ECFP6"]["n_cl"]
cl_cmap = plt.cm.get_cmap("tab20", max(n_cl,1))

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#F5F5F5")

noise_idx = [i for i,l in enumerate(labels) if l==-1]
if noise_idx:
    ax.scatter(emb[noise_idx,0], emb[noise_idx,1], c="#CCCCCC", s=40,
               alpha=0.5, label="Noise", zorder=2)

for cl in sorted(set(labels[labels>=0])):
    idx = [i for i,l in enumerate(labels) if l==cl]
    ax.scatter(emb[idx,0], emb[idx,1], c=[cl_cmap(cl)], s=60,
               alpha=0.85, edgecolors="white", linewidths=0.4,
               label=f"Cluster {cl+1} (n={len(idx)})", zorder=3)

ax.legend(fontsize=7, loc="best", framealpha=0.9, ncol=2)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title(f"VDR Ligand Chemical Space — Morgan ECFP6\nHDBSCAN Clustering ({n_cl} clusters)  |  n=153",
             fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.grid(alpha=0.2)
plt.tight_layout()
p = OUTDIR / "Fig8_umap_ecfp6_clusters.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 9 — UMAP labeled by ligand class ECFP6
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 9] UMAP labeled ECFP6 ...")
fig, ax = plt.subplots(figsize=(22, 15))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#F5F5F5")

for i, rec in enumerate(valid):
    color = CLASS_COLORS.get(rec["cls"], "#999")
    ax.scatter(emb[i,0], emb[i,1], c=color, s=55,
               alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
    ax.annotate(rec["code"], xy=(emb[i,0], emb[i,1]),
                xytext=(4,4), textcoords="offset points",
                fontsize=5.5, color="#111111", alpha=0.9, zorder=4)

legend = [mpatches.Patch(color=CLASS_COLORS[c],
          label=c.replace("_"," ").title()) for c in class_order
          if any(r["cls"]==c for r in valid)]
ax.legend(handles=legend, fontsize=11, loc="best", framealpha=0.9)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("VDR Ligand Chemical Space — Morgan ECFP6\nLabeled by Ligand Code, Colored by Class  |  n=153",
             fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.grid(alpha=0.2)
plt.tight_layout()
p = OUTDIR / "Fig9_umap_ecfp6_labeled.png"
plt.savefig(p, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 10 — UMAP by class ECFP4 (supplementary)
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 10] UMAP by class ECFP4 (supplementary) ...")
emb4   = fp_data["ECFP4"]["emb"]
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#F5F5F5")

for cls in class_order:
    idx = [i for i,r in enumerate(valid) if r["cls"]==cls]
    if idx:
        ax.scatter(emb4[idx,0], emb4[idx,1], c=CLASS_COLORS[cls], s=60,
                   alpha=0.85, edgecolors="white", linewidths=0.4,
                   label=cls.replace("_"," ").title(), zorder=3)

legend = [mpatches.Patch(color=CLASS_COLORS[c],
          label=c.replace("_"," ").title()) for c in class_order
          if any(r["cls"]==c for r in valid)]
ax.legend(handles=legend, fontsize=11, loc="best", framealpha=0.9)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("VDR Ligand Chemical Space — Morgan ECFP4\nColored by Ligand Class  |  n=153  (Supplementary)",
             fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.grid(alpha=0.2)
plt.tight_layout()
p = OUTDIR / "Fig10_umap_ecfp4_class_supplementary.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 11 — UMAP by cluster ECFP4 (supplementary)
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 11] UMAP by cluster ECFP4 (supplementary) ...")
labels4 = fp_data["ECFP4"]["labels"]
n_cl4   = fp_data["ECFP4"]["n_cl"]
cl_cmap4 = plt.cm.get_cmap("tab20", max(n_cl4,1))

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#F5F5F5")

noise4 = [i for i,l in enumerate(labels4) if l==-1]
if noise4:
    ax.scatter(emb4[noise4,0], emb4[noise4,1], c="#CCCCCC", s=40,
               alpha=0.5, label="Noise", zorder=2)
for cl in sorted(set(labels4[labels4>=0])):
    idx = [i for i,l in enumerate(labels4) if l==cl]
    ax.scatter(emb4[idx,0], emb4[idx,1], c=[cl_cmap4(cl)], s=60,
               alpha=0.85, edgecolors="white", linewidths=0.4,
               label=f"Cluster {cl+1} (n={len(idx)})", zorder=3)

ax.legend(fontsize=7, loc="best", framealpha=0.9, ncol=2)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title(f"VDR Ligand Chemical Space — Morgan ECFP4\nHDBSCAN Clustering ({n_cl4} clusters)  |  n=153  (Supplementary)",
             fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.grid(alpha=0.2)
plt.tight_layout()
p = OUTDIR / "Fig11_umap_ecfp4_clusters_supplementary.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL FIGURES GENERATED")
print("="*60)
figs = [
    ("Fig1",  "Class distribution bar chart"),
    ("Fig2",  "Descriptor histograms by class"),
    ("Fig3",  "Lipinski pass/fail by class"),
    ("Fig4",  "Tanimoto heatmap ECFP4"),
    ("Fig5",  "Tanimoto heatmap ECFP6"),
    ("Fig6",  "UMAP by class ECFP6  ← MAIN FIGURE"),
    ("Fig7",  "UMAP by species ECFP6"),
    ("Fig8",  "UMAP by cluster ECFP6"),
    ("Fig9",  "UMAP labeled ECFP6"),
    ("Fig10", "UMAP by class ECFP4  (supplementary)"),
    ("Fig11", "UMAP by cluster ECFP4  (supplementary)"),
]
for code, desc in figs:
    print(f"  {code}: {desc}")
print(f"\nAll saved to: {OUTDIR}")
