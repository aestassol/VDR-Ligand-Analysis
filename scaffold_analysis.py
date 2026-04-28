#!/usr/bin/env python3
"""
Bemis-Murcko Scaffold Analysis for VDR Ligands
================================================
Input:  vdr_ligands_final.csv
Output:
  Fig22_scaffold_distribution.png  — top scaffolds by frequency
  Fig23_scaffold_by_class.png      — scaffold diversity per ligand class
  Fig24_scaffold_umap.png          — UMAP colored by scaffold family
  vdr_scaffold_results.csv         — per-ligand scaffold SMILES + family

Bemis-Murcko framework = core ring system stripped of side chains
Generic scaffold = all atoms replaced by carbons (shape-only)
"""

import csv, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("conda install -c conda-forge rdkit")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
except ImportError:
    raise SystemExit("pip install matplotlib")

try:
    import umap as umap_lib
except ImportError:
    raise SystemExit("pip install umap-learn")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT  = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_final.csv")
OUTDIR = INPUT.parent

BORON_CODES = {"A1MAV","A1MAW","A1MAX","A1MAY","A1MAZ","A1MAU","M7E"}

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

# ── Load ──────────────────────────────────────────────────────────────────────
with open(INPUT, newline="", encoding="utf-8") as fh:
    sample = fh.read(2000); fh.seek(0)
    delim  = ";" if sample.count(";") > sample.count(",") else ","
    rows   = list(csv.DictReader(fh, delimiter=delim))

records = []
for row in rows:
    code   = row["PyMOL_Ligand_Name"].strip()
    smiles = row.get("SMILES_stereo","").strip() or row.get("SMILES","").strip()
    mol    = None
    if code not in BORON_CODES and smiles:
        mol = Chem.MolFromSmiles(smiles)
    if mol:
        records.append({
            "code":    code,
            "species": row["Species"].strip(),
            "cls":     row["Ligand_Class"].strip(),
            "mol":     mol,
            "smiles":  smiles,
        })

print(f"Valid molecules: {len(records)}")

# ── Compute Bemis-Murcko scaffolds ────────────────────────────────────────────
print("Computing Bemis-Murcko scaffolds ...")

for rec in records:
    try:
        # Full scaffold (with heteroatoms)
        scaffold     = MurckoScaffold.GetScaffoldForMol(rec["mol"])
        scaffold_smi = Chem.MolToSmiles(scaffold) if scaffold else ""

        # Generic scaffold (carbon-only, shape only)
        generic      = MurckoScaffold.MakeScaffoldGeneric(scaffold) if scaffold else None
        generic_smi  = Chem.MolToSmiles(generic) if generic else ""

        rec["scaffold_smi"] = scaffold_smi
        rec["generic_smi"]  = generic_smi
        rec["scaffold_mol"] = scaffold
        rec["generic_mol"]  = generic

    except Exception:
        rec["scaffold_smi"] = ""
        rec["generic_smi"]  = ""
        rec["scaffold_mol"] = None
        rec["generic_mol"]  = None

# ── Count scaffolds ───────────────────────────────────────────────────────────
scaffold_counter = Counter(r["scaffold_smi"] for r in records if r["scaffold_smi"])
generic_counter  = Counter(r["generic_smi"]  for r in records if r["generic_smi"])

# Assign scaffold family label (top N get own label, rest = "Other")
TOP_N = 12
top_scaffolds = [s for s, _ in scaffold_counter.most_common(TOP_N)]
scaffold_to_family = {}
for i, smi in enumerate(top_scaffolds):
    scaffold_to_family[smi] = f"Scaffold {i+1} (n={scaffold_counter[smi]})"

for rec in records:
    smi = rec["scaffold_smi"]
    rec["scaffold_family"] = scaffold_to_family.get(smi, f"Other (n=1)")

print(f"\nUnique scaffolds (Murcko):  {len(scaffold_counter)}")
print(f"Unique scaffolds (generic): {len(generic_counter)}")
print(f"\nTop 12 scaffolds:")
for i, (smi, n) in enumerate(scaffold_counter.most_common(12), 1):
    # Find which classes use this scaffold
    classes = [r["cls"] for r in records if r["scaffold_smi"]==smi]
    cls_str = ", ".join(f"{c}:{v}" for c,v in Counter(classes).most_common())
    print(f"  {i:2}. n={n:3}  {cls_str}")
    print(f"      {smi[:80]}")

# ═════════════════════════════════════════════════════════════════════════════
# Fig 22 — Scaffold frequency distribution
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Fig 22] Scaffold distribution ...")

top20 = scaffold_counter.most_common(20)
labels_20 = []
counts_20 = []
colors_20 = []

for smi, n in top20:
    classes = [r["cls"] for r in records if r["scaffold_smi"]==smi]
    dominant = Counter(classes).most_common(1)[0][0]
    labels_20.append(f"S{top_scaffolds.index(smi)+1}" if smi in top_scaffolds else "?")
    counts_20.append(n)
    colors_20.append(CLASS_COLORS.get(dominant, "#999"))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#FAFAFA")
fig.suptitle("Bemis-Murcko Scaffold Analysis — VDR Ligands\nn=153 molecules",
             fontsize=13, fontweight="bold")

# Left: top 20 scaffold bar chart
ax = axes[0]
ax.set_facecolor("#F5F5F5")
y = range(len(counts_20))
bars = ax.barh(list(y), counts_20, color=colors_20,
               edgecolor="white", alpha=0.85)
ax.set_yticks(list(y))
ax.set_yticklabels([f"Scaffold {i+1}  (n={n})"
                    for i,(smi,n) in enumerate(top20)], fontsize=8)
ax.set_xlabel("Number of ligands", fontsize=11)
ax.set_title("Top 20 Most Common Scaffolds\n(colored by dominant ligand class)",
             fontsize=10, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)

legend_handles = [mpatches.Patch(color=CLASS_COLORS[c],
                  label=c.replace("_"," ").title())
                  for c in CLASS_COLORS if c != "boron_cluster"]
ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

# Right: scaffold count summary
ax2 = axes[1]
ax2.set_facecolor("#F5F5F5")

total     = len(records)
unique_sc = len(scaffold_counter)
singleton = sum(1 for n in scaffold_counter.values() if n==1)
top1_n    = scaffold_counter.most_common(1)[0][1]
top5_n    = sum(n for _,n in scaffold_counter.most_common(5))

categories = ["Total\nligands", "Unique\nscaffolds",
              "Singleton\nscaffolds", "Top 5 scaffolds\ncoverage"]
values     = [total, unique_sc, singleton, top5_n]
bar_colors = ["#1565C0","#E65100","#9E9E9E","#2E7D32"]

bars2 = ax2.bar(categories, values, color=bar_colors,
                edgecolor="white", alpha=0.85, width=0.5)
for bar, val in zip(bars2, values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             str(val), ha="center", va="bottom",
             fontsize=12, fontweight="bold")

ax2.set_ylabel("Count", fontsize=11)
ax2.set_title("Scaffold Diversity Summary", fontsize=10, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
p = OUTDIR / "Fig22_scaffold_distribution.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# Fig 23 — Scaffold diversity per class
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 23] Scaffold diversity by class ...")

class_order = ["secosteroid","non_steroidal","steroidal","gemini"]
fig, axes   = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#FAFAFA")
fig.suptitle("Scaffold Diversity by Ligand Class",
             fontsize=13, fontweight="bold")

# Left: unique scaffolds per class
ax = axes[0]
ax.set_facecolor("#F5F5F5")
unique_per_class = []
total_per_class  = []
for cls in class_order:
    cls_recs = [r for r in records if r["cls"]==cls]
    unique_per_class.append(len(set(r["scaffold_smi"] for r in cls_recs)))
    total_per_class.append(len(cls_recs))

x     = np.arange(len(class_order))
w     = 0.35
bars1 = ax.bar(x-w/2, total_per_class,  w, label="Total ligands",
               color=[CLASS_COLORS[c] for c in class_order],
               alpha=0.5, edgecolor="white")
bars2 = ax.bar(x+w/2, unique_per_class, w, label="Unique scaffolds",
               color=[CLASS_COLORS[c] for c in class_order],
               alpha=0.95, edgecolor="white")

for bar, n in zip(bars1, total_per_class):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            str(n), ha="center", fontsize=10, fontweight="bold")
for bar, n in zip(bars2, unique_per_class):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            str(n), ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_"," ").title() for c in class_order],
                   fontsize=10, rotation=15, ha="right")
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Total Ligands vs Unique Scaffolds per Class\n(light=total, dark=unique)",
             fontsize=10, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# Right: scaffold diversity ratio (unique/total)
ax2 = axes[1]
ax2.set_facecolor("#F5F5F5")
ratios = [u/t*100 if t>0 else 0 for u,t in zip(unique_per_class, total_per_class)]
bars3  = ax2.bar([c.replace("_"," ").title() for c in class_order],
                 ratios,
                 color=[CLASS_COLORS[c] for c in class_order],
                 edgecolor="white", alpha=0.85, width=0.5)
for bar, r in zip(bars3, ratios):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{r:.0f}%", ha="center", fontsize=11, fontweight="bold")

ax2.set_ylabel("Scaffold diversity (%)", fontsize=11)
ax2.set_title("Scaffold Diversity Ratio\n(unique scaffolds / total ligands × 100)",
              fontsize=10, fontweight="bold")
ax2.set_xticklabels([c.replace("_"," ").title() for c in class_order],
                    fontsize=10, rotation=15, ha="right")
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, 115)

plt.tight_layout()
p = OUTDIR / "Fig23_scaffold_by_class.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ═════════════════════════════════════════════════════════════════════════════
# Fig 24 — UMAP colored by scaffold family
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 24] UMAP by scaffold family ...")

fps = [AllChem.GetMorganFingerprintAsBitVect(r["mol"], 3, nBits=2048)
       for r in records]
arr = np.zeros((len(fps), 2048), dtype=np.uint8)
for i, fp in enumerate(fps):
    DataStructs.ConvertToNumpyArray(fp, arr[i])

reducer  = umap_lib.UMAP(n_components=2, n_neighbors=15,
                          min_dist=0.1, metric="jaccard", random_state=42)
emb      = reducer.fit_transform(arr)

# Color by scaffold family (top 8 + other)
top8_scaffolds = [s for s,_ in scaffold_counter.most_common(8)]
cmap = plt.cm.get_cmap("tab10", 9)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor("#FAFAFA")
fig.suptitle("VDR Ligand Chemical Space — Colored by Scaffold Family\nMorgan ECFP6 UMAP  |  n=153",
             fontsize=13, fontweight="bold")

# Left: by scaffold family
ax1 = axes[0]
ax1.set_facecolor("#F5F5F5")
legend_handles = []

for i, smi in enumerate(top8_scaffolds):
    idx = [j for j,r in enumerate(records) if r["scaffold_smi"]==smi]
    n   = len(idx)
    # Get dominant class for label
    dom_cls = Counter(records[j]["cls"] for j in idx).most_common(1)[0][0]
    label   = f"S{i+1}: {dom_cls.replace('_',' ')} (n={n})"
    color   = cmap(i)
    ax1.scatter(emb[idx,0], emb[idx,1], c=[color], s=60,
                alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
    legend_handles.append(mpatches.Patch(color=color, label=label))

# Other scaffolds
other_idx = [j for j,r in enumerate(records)
             if r["scaffold_smi"] not in top8_scaffolds]
if other_idx:
    ax1.scatter(emb[other_idx,0], emb[other_idx,1], c="#CCCCCC",
                s=40, alpha=0.5, edgecolors="white", linewidths=0.3,
                label="Other scaffolds", zorder=2)
    legend_handles.append(mpatches.Patch(color="#CCCCCC",
                           label=f"Other (n={len(other_idx)})"))

ax1.legend(handles=legend_handles, fontsize=7, loc="best",
           framealpha=0.9, title="Top 8 Scaffolds")
ax1.set_xlabel("UMAP 1", fontsize=11)
ax1.set_ylabel("UMAP 2", fontsize=11)
ax1.set_title("By Scaffold Family", fontsize=11, fontweight="bold")
ax1.spines[["top","right"]].set_visible(False)
ax1.grid(alpha=0.2)

# Right: by ligand class (reference)
ax2 = axes[1]
ax2.set_facecolor("#F5F5F5")
for cls in ["secosteroid","non_steroidal","steroidal","gemini"]:
    idx = [j for j,r in enumerate(records) if r["cls"]==cls]
    if idx:
        ax2.scatter(emb[idx,0], emb[idx,1],
                    c=CLASS_COLORS[cls], s=60, alpha=0.85,
                    edgecolors="white", linewidths=0.4,
                    label=cls.replace("_"," ").title(), zorder=3)

ax2.legend(fontsize=10, loc="best", framealpha=0.9)
ax2.set_xlabel("UMAP 1", fontsize=11)
ax2.set_ylabel("UMAP 2", fontsize=11)
ax2.set_title("By Ligand Class (reference)", fontsize=11, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(alpha=0.2)

plt.tight_layout()
p = OUTDIR / "Fig24_scaffold_umap.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {p.name}")

# ── Save scaffold CSV ─────────────────────────────────────────────────────────
print("\nSaving scaffold CSV ...")
out_csv = OUTDIR / "vdr_scaffold_results.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as fh:
    cols = ["PyMOL_Ligand_Name","Species","Ligand_Class",
            "Scaffold_SMILES","Generic_Scaffold_SMILES","Scaffold_Family"]
    writer = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
    writer.writeheader()
    for rec in records:
        writer.writerow({
            "PyMOL_Ligand_Name":      rec["code"],
            "Species":                rec["species"],
            "Ligand_Class":           rec["cls"],
            "Scaffold_SMILES":        rec["scaffold_smi"],
            "Generic_Scaffold_SMILES":rec["generic_smi"],
            "Scaffold_Family":        rec["scaffold_family"],
        })

print(f"  Saved: {out_csv.name}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SCAFFOLD ANALYSIS COMPLETE")
print("="*55)
print(f"  Total ligands:          {len(records)}")
print(f"  Unique Murcko scaffolds:{len(scaffold_counter)}")
print(f"  Unique generic scaffolds:{len(generic_counter)}")
print(f"  Singleton scaffolds:    {singleton}")
print(f"  Top scaffold coverage:  {top1_n} ligands share scaffold 1")
print()
print("Scaffold diversity by class:")
for cls, u, t in zip(class_order, unique_per_class, total_per_class):
    print(f"  {cls:<20} {u:3} unique / {t:3} total  ({u/t*100:.0f}%)")
print()
print("Figures saved:")
for i in [22, 23, 24]:
    print(f"  Fig{i}_scaffold_*.png")
print(f"  vdr_scaffold_results.csv")
