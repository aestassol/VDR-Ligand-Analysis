#!/usr/bin/env python3
"""
Full analysis pipeline on vdr_ligands_final.csv
Steps:
  1. Lipinski / Veber descriptors (recomputed)
  2. Morgan fingerprints ECFP4 + ECFP6
  3. UMAP + HDBSCAN clustering
  4. Tanimoto heatmap
  5. Labeled UMAP

Outputs (all in same folder as input):
  vdr_final_lipinski.csv
  vdr_final_fingerprints.csv
  vdr_final_umap_ecfp4.png
  vdr_final_umap_ecfp6.png
  vdr_final_umap_ecfp4_labeled.png
  vdr_final_umap_ecfp6_labeled.png
  vdr_final_heatmap_ecfp4.png
  vdr_final_heatmap_ecfp6.png
"""

import csv, warnings, math
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("conda install -c conda-forge rdkit")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
except ImportError:
    raise SystemExit("pip install matplotlib")

try:
    import umap
except ImportError:
    raise SystemExit("pip install umap-learn")

try:
    import hdbscan
except ImportError:
    raise SystemExit("pip install hdbscan")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis")
INPUT = BASE / "vdr_ligands_final.csv"

# ── Colors ────────────────────────────────────────────────────────────────────
SPECIES_COLORS = {
    "Human":     "#2196F3",
    "Rat":       "#FF5722",
    "Zebrafish": "#4CAF50",
}
CLASS_COLORS = {
    "secosteroid":  "#1565C0",
    "non_steroidal":"#E65100",
    "steroidal":    "#2E7D32",
    "gemini":       "#6A1B9A",
    "boron_cluster":"#BBBBBB",
}

BORON_CODES = {"A1MAV","A1MAW","A1MAX","A1MAY","A1MAZ","A1MAU","M7E"}

# ── Load ──────────────────────────────────────────────────────────────────────
with open(INPUT, newline="", encoding="utf-8") as fh:
    sample = fh.read(2000); fh.seek(0)
    delim = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(fh, delimiter=delim)
    fieldnames = list(reader.fieldnames)
    all_rows = list(reader)

print(f"Loaded {len(all_rows)} rows from {INPUT.name}")

# ── STEP 1: Lipinski / Veber ──────────────────────────────────────────────────
print("\n[1] Computing Lipinski / Veber descriptors ...")

def compute_props(smiles, code):
    empty = dict(MW="", HBD="", HBA="", LogP="", RotBonds="", TPSA="",
                 Lipinski_violations="", Lipinski_pass="", Veber_pass="", Druglike_pass="")
    if code in BORON_CODES:
        empty.update(Lipinski_pass="boron_cluster_excluded",
                     Veber_pass="boron_cluster_excluded",
                     Druglike_pass="boron_cluster_excluded")
        return empty
    if not smiles.strip():
        return empty
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return empty
    mw       = round(Descriptors.ExactMolWt(mol), 2)
    hbd      = rdMolDescriptors.CalcNumHBD(mol)
    hba      = rdMolDescriptors.CalcNumHBA(mol)
    logp     = round(Descriptors.MolLogP(mol), 2)
    rotb     = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa     = round(rdMolDescriptors.CalcTPSA(mol), 2)
    viols    = sum([mw>500, hbd>5, hba>10, logp>5])
    lip      = "PASS" if viols <= 1 else "FAIL"
    veb      = "PASS" if (rotb<=10 and tpsa<=140) else "FAIL"
    drug     = "PASS" if lip=="PASS" and veb=="PASS" else "FAIL"
    return dict(MW=mw, HBD=hbd, HBA=hba, LogP=logp, RotBonds=rotb, TPSA=tpsa,
                Lipinski_violations=viols, Lipinski_pass=lip, Veber_pass=veb, Druglike_pass=drug)

# Remove old descriptor cols, recompute
desc_cols = ["MW","HBD","HBA","LogP","RotBonds","TPSA",
             "Lipinski_violations","Lipinski_pass","Veber_pass","Druglike_pass"]
base_cols = [c for c in fieldnames if c not in desc_cols]
new_fieldnames = base_cols + desc_cols

lip_rows = []
for row in all_rows:
    code   = row["PyMOL_Ligand_Name"].strip()
    smiles = row.get("SMILES_stereo","").strip() or row.get("SMILES","").strip()
    props  = compute_props(smiles, code)
    new_row = {c: row.get(c,"") for c in base_cols}
    new_row.update(props)
    lip_rows.append(new_row)

lip_out = BASE / "vdr_final_lipinski.csv"
with open(lip_out, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=new_fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(lip_rows)

lip_pass  = sum(1 for r in lip_rows if r["Lipinski_pass"]=="PASS")
veb_pass  = sum(1 for r in lip_rows if r["Veber_pass"]=="PASS")
drug_pass = sum(1 for r in lip_rows if r["Druglike_pass"]=="PASS")
print(f"  Lipinski PASS: {lip_pass}/160  Veber PASS: {veb_pass}/160  Druglike: {drug_pass}/160")
print(f"  Saved: {lip_out.name}")

# ── STEP 2: Morgan fingerprints + UMAP + HDBSCAN ──────────────────────────────
print("\n[2] Computing Morgan fingerprints ...")

valid, boron_rec = [], []
for row in lip_rows:
    code   = row["PyMOL_Ligand_Name"].strip()
    smiles = row.get("SMILES_stereo","").strip() or row.get("SMILES","").strip()
    mol    = None if code in BORON_CODES else Chem.MolFromSmiles(smiles) if smiles else None
    rec    = {"row": row, "code": code, "species": row["Species"],
              "lig_class": row["Ligand_Class"], "mol": mol}
    (boron_rec if mol is None else valid).append(rec)

print(f"  Valid: {len(valid)}  Boron/excluded: {len(boron_rec)}")

def fps_to_matrix(fps):
    arr = np.zeros((len(fps), len(fps[0])), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr

def tanimoto_matrix(fps):
    n   = len(fps)
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    return mat

fp_results = {}
for radius in [2, 3]:
    ecfp = f"ECFP{radius*2}"
    print(f"\n  [{ecfp}] Computing fingerprints + UMAP + HDBSCAN ...")
    fps    = [AllChem.GetMorganFingerprintAsBitVect(r["mol"], radius, nBits=2048) for r in valid]
    fp_mat = fps_to_matrix(fps)
    tan_mat= tanimoto_matrix(fps)

    reducer  = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                         metric="jaccard", random_state=42)
    embedding = reducer.fit_transform(fp_mat)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric="euclidean")
    labels    = clusterer.fit_predict(embedding)

    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"    Clusters: {n_cl}  Noise: {sum(1 for l in labels if l==-1)}")

    fp_results[ecfp] = dict(fps=fps, fp_mat=fp_mat, tan_mat=tan_mat,
                             embedding=embedding, labels=labels)

# ── STEP 3: Save fingerprint CSV ──────────────────────────────────────────────
print("\n[3] Saving fingerprint CSV ...")
fp_csv = BASE / "vdr_final_fingerprints.csv"
with open(fp_csv, "w", newline="", encoding="utf-8") as fh:
    cols = ["Protein_PDB","PyMOL_Ligand_Name","Correct_CCD_Code","Species",
            "Ligand_Class","Lipinski_pass","InChIKey",
            "UMAP1_ECFP4","UMAP2_ECFP4","Cluster_ECFP4",
            "UMAP1_ECFP6","UMAP2_ECFP6","Cluster_ECFP6"]
    writer = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
    writer.writeheader()
    emb4  = fp_results["ECFP4"]["embedding"]
    cl4   = fp_results["ECFP4"]["labels"]
    emb6  = fp_results["ECFP6"]["embedding"]
    cl6   = fp_results["ECFP6"]["labels"]
    for i, rec in enumerate(valid):
        cl4_label = "noise" if cl4[i]==-1 else f"Cluster_{int(cl4[i])+1}"
        cl6_label = "noise" if cl6[i]==-1 else f"Cluster_{int(cl6[i])+1}"
        writer.writerow({
            "Protein_PDB":       rec["row"].get("Protein_PDB",""),
            "PyMOL_Ligand_Name": rec["code"],
            "Correct_CCD_Code":  rec["row"].get("Correct_CCD_Code",""),
            "Species":           rec["species"],
            "Ligand_Class":      rec["lig_class"],
            "Lipinski_pass":     rec["row"].get("Lipinski_pass",""),
            "InChIKey":          rec["row"].get("InChIKey",""),
            "UMAP1_ECFP4": round(float(emb4[i,0]),4),
            "UMAP2_ECFP4": round(float(emb4[i,1]),4),
            "Cluster_ECFP4":     cl4_label,
            "UMAP1_ECFP6": round(float(emb6[i,0]),4),
            "UMAP2_ECFP6": round(float(emb6[i,1]),4),
            "Cluster_ECFP6":     cl6_label,
        })
print(f"  Saved: {fp_csv.name}")

# ── STEP 4: UMAP plots ────────────────────────────────────────────────────────
print("\n[4] Generating UMAP plots ...")

def plot_umap(embedding, labels, records, radius, outpath, outpath_labeled):
    n_cl      = len(set(labels)) - (1 if -1 in labels else 0)
    cl_cmap   = plt.cm.get_cmap("tab20", max(n_cl,1))

    for mode in ["species", "class", "cluster"]:
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#F5F5F5")

        if mode == "species":
            for sp, color in SPECIES_COLORS.items():
                idx = [i for i,r in enumerate(records) if r["species"]==sp]
                if idx:
                    ax.scatter(embedding[idx,0], embedding[idx,1], c=color,
                               s=60, alpha=0.85, edgecolors="white", linewidths=0.4,
                               label=sp, zorder=3)
            legend = [mpatches.Patch(color=SPECIES_COLORS[s], label=s)
                      for s in ["Human","Rat","Zebrafish"]]
            title_suffix = "By Species"

        elif mode == "class":
            for cls, color in CLASS_COLORS.items():
                idx = [i for i,r in enumerate(records) if r["lig_class"]==cls]
                if idx:
                    ax.scatter(embedding[idx,0], embedding[idx,1], c=color,
                               s=60, alpha=0.85, edgecolors="white", linewidths=0.4,
                               label=cls, zorder=3)
            legend = [mpatches.Patch(color=CLASS_COLORS[c], label=c)
                      for c in CLASS_COLORS if any(r["lig_class"]==c for r in records)]
            title_suffix = "By Ligand Class"

        else:  # cluster
            noise = [i for i,l in enumerate(labels) if l==-1]
            if noise:
                ax.scatter(embedding[noise,0], embedding[noise,1],
                           c="#CCCCCC", s=40, alpha=0.5, label="noise", zorder=2)
            for cl in sorted(set(labels[labels>=0])):
                idx = [i for i,l in enumerate(labels) if l==cl]
                ax.scatter(embedding[idx,0], embedding[idx,1],
                           c=[cl_cmap(cl)], s=60, alpha=0.85,
                           edgecolors="white", linewidths=0.4,
                           label=f"Cluster {cl+1} (n={len(idx)})", zorder=3)
            legend = None
            title_suffix = f"By HDBSCAN Cluster ({n_cl} clusters)"

        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.set_title(f"VDR Ligands — Morgan ECFP{radius*2}\n{title_suffix}  |  n={len(records)}",
                     fontsize=12, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=0.2)
        if legend:
            ax.legend(handles=legend, fontsize=9, loc="best", framealpha=0.9)
        else:
            ax.legend(fontsize=7, loc="best", framealpha=0.9,
                      ncol=2 if n_cl>8 else 1)

        suffix = {"species":"species","class":"class","cluster":"cluster"}[mode]
        p = outpath.parent / f"{outpath.stem}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"    Saved: {p.name}")

    # Labeled plot
    fig, ax = plt.subplots(figsize=(22, 16))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")
    for i, rec in enumerate(records):
        color = CLASS_COLORS.get(rec["lig_class"], "#999999")
        ax.scatter(embedding[i,0], embedding[i,1], c=color, s=55,
                   alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
        ax.annotate(rec["code"], xy=(embedding[i,0], embedding[i,1]),
                    xytext=(4,4), textcoords="offset points",
                    fontsize=5.5, color="#222222", alpha=0.9, zorder=4)
    legend = [mpatches.Patch(color=CLASS_COLORS[c], label=c)
              for c in CLASS_COLORS if any(r["lig_class"]==c for r in records)]
    ax.legend(handles=legend, fontsize=10, loc="best", framealpha=0.9)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title(f"VDR Ligands — Morgan ECFP{radius*2} — Labeled by Ligand Class\nn={len(records)}",
                 fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(outpath_labeled, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved: {outpath_labeled.name}")


for radius in [2, 3]:
    ecfp = f"ECFP{radius*2}"
    plot_umap(
        fp_results[ecfp]["embedding"],
        fp_results[ecfp]["labels"],
        valid, radius,
        BASE / f"vdr_final_umap_ecfp{radius*2}",
        BASE / f"vdr_final_umap_ecfp{radius*2}_labeled.png",
    )

# ── STEP 5: Tanimoto heatmaps ─────────────────────────────────────────────────
print("\n[5] Generating Tanimoto heatmaps ...")

def plot_heatmap(tan_mat, records, radius, outpath):
    n      = len(records)
    # Sort by ligand class then species
    order  = sorted(range(n), key=lambda i: (records[i]["lig_class"], records[i]["species"]))
    mat_s  = tan_mat[np.ix_(order, order)]
    labels = [records[i]["code"] for i in order]
    classes= [records[i]["lig_class"] for i in order]

    row_colors = [CLASS_COLORS.get(c, "#999999") for c in classes]

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor("#FAFAFA")
    im = ax.imshow(mat_s, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Tanimoto Similarity")

    for i, color in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-2.8, i-0.5), 2.2, 1,
                                   color=color, clip_on=False))

    if n <= 60:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Ligands (sorted by class)", fontsize=10)
        ax.set_ylabel("Ligands (sorted by class)", fontsize=10)

    ax.set_title(f"Tanimoto Similarity — Morgan ECFP{radius*2}\n"
                 f"n={n}, sorted by ligand class", fontsize=12, fontweight="bold")

    legend = [mpatches.Patch(color=CLASS_COLORS[c], label=c)
              for c in CLASS_COLORS if c in set(classes)]
    ax.legend(handles=legend, loc="upper right",
              bbox_to_anchor=(1.22, 1.0), fontsize=9, title="Class")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved: {outpath.name}")

for radius in [2, 3]:
    ecfp = f"ECFP{radius*2}"
    plot_heatmap(fp_results[ecfp]["tan_mat"], valid, radius,
                 BASE / f"vdr_final_heatmap_ecfp{radius*2}.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"  vdr_final_lipinski.csv")
print(f"  vdr_final_fingerprints.csv")
print(f"  vdr_final_umap_ecfp4_species/class/cluster.png")
print(f"  vdr_final_umap_ecfp6_species/class/cluster.png")
print(f"  vdr_final_umap_ecfp4_labeled.png")
print(f"  vdr_final_umap_ecfp6_labeled.png")
print(f"  vdr_final_heatmap_ecfp4.png")
print(f"  vdr_final_heatmap_ecfp6.png")
print(f"\nClass distribution:")
for cls, n in sorted(Counter(r["Ligand_Class"] for r in lip_rows).items(), key=lambda x:-x[1]):
    print(f"  {cls:<20} {n}")
