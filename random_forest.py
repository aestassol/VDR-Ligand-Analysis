#!/usr/bin/env python3
"""
VDR Ligand Random Forest Classification
=========================================
Model 1: Predict Ligand Class  — features: ECFP6 Morgan fingerprints
Model 2: Predict Species        — features: physicochemical descriptors

Output:
  Fig16_rf_class_confusion.png      — confusion matrix (class)
  Fig17_rf_class_importance.png     — feature importance (class)
  Fig18_rf_class_roc.png            — ROC curves (class)
  Fig19_rf_species_confusion.png    — confusion matrix (species)
  Fig20_rf_species_importance.png   — feature importance (species)
  Fig21_rf_species_roc.png          — ROC curves (species)
  vdr_rf_results.csv                — per-sample predictions + probabilities
"""

import csv, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("conda install -c conda-forge rdkit")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, roc_curve, auc
    )
    from sklearn.preprocessing import StandardScaler, label_binarize
except ImportError:
    raise SystemExit("pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
except ImportError:
    raise SystemExit("pip install matplotlib seaborn")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT  = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_final.csv")
OUTDIR = INPUT.parent

BORON_CODES = {"A1MAV","A1MAW","A1MAX","A1MAY","A1MAZ","A1MAU","M7E"}

CLASS_COLORS = {
    "secosteroid":   "#1565C0",
    "non_steroidal": "#E65100",
    "steroidal":     "#2E7D32",
    "gemini":        "#6A1B9A",
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
        })

print(f"Valid molecules: {len(records)}")

# ── Feature computation ───────────────────────────────────────────────────────
# ECFP6 fingerprints
fps = [AllChem.GetMorganFingerprintAsBitVect(r["mol"], 3, nBits=2048) for r in records]
fp_arr = np.zeros((len(fps), 2048), dtype=np.uint8)
for i, fp in enumerate(fps):
    DataStructs.ConvertToNumpyArray(fp, fp_arr[i])

# Physicochemical descriptors
DESC_NAMES = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]
def get_desc(mol):
    return [
        Descriptors.ExactMolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
    ]

desc_arr = np.array([get_desc(r["mol"]) for r in records])
desc_sc  = StandardScaler().fit_transform(desc_arr)

# Labels
# For class: exclude gemini (n=1, cannot be cross-validated)
cls_labels  = [r["cls"] for r in records]
sp_labels   = [r["species"] for r in records]

# Filter gemini out of class model
mask_cls = [i for i,r in enumerate(records) if r["cls"] != "gemini"]
fp_cls   = fp_arr[mask_cls]
y_cls    = [cls_labels[i] for i in mask_cls]
rec_cls  = [records[i] for i in mask_cls]

class_order   = ["secosteroid","non_steroidal","steroidal"]
species_order = ["Human","Rat","Zebrafish"]

print(f"Class model:   {len(y_cls)} samples  {Counter(y_cls)}")
print(f"Species model: {len(sp_labels)} samples  {Counter(sp_labels)}")

# ── Random Forest + cross-validation ─────────────────────────────────────────
def run_rf(X, y, label_order, n_estimators=300, cv=5, random_state=42):
    rf  = RandomForestClassifier(n_estimators=n_estimators,
                                  max_features="sqrt",
                                  class_weight="balanced",
                                  random_state=random_state,
                                  n_jobs=-1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    y_pred  = cross_val_predict(rf, X, y, cv=skf, method="predict")
    y_proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")

    acc = accuracy_score(y, y_pred)
    cm  = confusion_matrix(y, y_pred, labels=label_order)
    rep = classification_report(y, y_pred, labels=label_order,
                                 target_names=label_order, output_dict=True)

    # Fit on full data for feature importance
    rf.fit(X, y)
    importance = rf.feature_importances_

    print(f"  Accuracy: {acc*100:.1f}%")
    return acc, cm, rep, y_pred, y_proba, importance


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_confusion(cm, labels, colors, title, outpath, acc):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.04, label="Proportion")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val     = cm[i, j]
            val_norm= cm_norm[i, j]
            ax.text(j, i, f"{val}\n({val_norm*100:.0f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if val_norm > 0.6 else "black")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    clean = [l.replace("_"," ").title() for l in labels]
    ax.set_xticklabels(clean, fontsize=11, rotation=20, ha="right")
    ax.set_yticklabels(clean, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{title}\n5-fold CV accuracy: {acc*100:.1f}%",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {outpath.name}")


def plot_roc(X, y, label_order, colors, title, outpath, random_state=42):
    rf  = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                  class_weight="balanced",
                                  random_state=random_state, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y_proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")

    y_bin = label_binarize(y, classes=label_order)
    # Get class indices from RF fit
    rf.fit(X, y)
    cls_idx = {c: i for i, c in enumerate(rf.classes_)}

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    for i, label in enumerate(label_order):
        if label not in cls_idx:
            continue
        col_idx = cls_idx[label]
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, col_idx])
        roc_auc     = auc(fpr, tpr)
        clean_label = label.replace("_"," ").title()
        ax.plot(fpr, tpr, color=colors.get(label,"#999"),
                linewidth=2.5, label=f"{clean_label} (AUC={roc_auc:.3f})")

    ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{title}\nROC Curves — 5-fold Cross-Validation",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {outpath.name}")


def plot_importance(importance, feat_names, top_n, colors, title, outpath):
    idx     = np.argsort(importance)[-top_n:]
    vals    = importance[idx]
    names   = [feat_names[i] for i in idx]
    bar_colors = [colors[i % len(colors)] for i in range(len(idx))]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n*0.35)))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    bars = ax.barh(range(len(idx)), vals, color=bar_colors,
                   edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {outpath.name}")


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1: Ligand Class — ECFP6 fingerprints
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Model 1] Ligand class prediction — ECFP6 fingerprints ...")
acc1, cm1, rep1, pred1, proba1, imp1 = run_rf(
    fp_cls, y_cls, class_order
)

# Confusion matrix
plot_confusion(
    cm1, class_order, CLASS_COLORS,
    "Ligand Class Prediction — Random Forest (ECFP6)",
    OUTDIR / "Fig16_rf_class_confusion.png", acc1
)

# Feature importance — top 25 fingerprint bits
bit_names = [f"Bit_{i}" for i in range(2048)]
plot_importance(
    imp1, bit_names, top_n=25,
    colors=["#1565C0","#1976D2","#1E88E5","#2196F3","#42A5F5"],
    title="Top 25 ECFP6 Fingerprint Bits — Ligand Class Prediction",
    outpath=OUTDIR / "Fig17_rf_class_importance.png"
)

# ROC curves
plot_roc(
    fp_cls, y_cls, class_order, CLASS_COLORS,
    "Ligand Class Prediction — Random Forest (ECFP6)",
    OUTDIR / "Fig18_rf_class_roc.png"
)

# Classification report
print(f"\n  Classification report (Ligand Class):")
for cls in class_order:
    m = rep1.get(cls, {})
    print(f"    {cls:<20}  P={m.get('precision',0):.3f}  "
          f"R={m.get('recall',0):.3f}  F1={m.get('f1-score',0):.3f}  "
          f"n={m.get('support',0):.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2: Species — physicochemical descriptors
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Model 2] Species prediction — physicochemical descriptors ...")
acc2, cm2, rep2, pred2, proba2, imp2 = run_rf(
    desc_sc, sp_labels, species_order
)

# Confusion matrix
plot_confusion(
    cm2, species_order, SPECIES_COLORS,
    "Species Prediction — Random Forest (Physicochemical Descriptors)",
    OUTDIR / "Fig19_rf_species_confusion.png", acc2
)

# Feature importance — all 6 descriptors
plot_importance(
    imp2, DESC_NAMES, top_n=6,
    colors=["#E65100","#EF6C00","#F57C00","#FB8C00","#FFA726","#FFB74D"],
    title="Descriptor Importance — Species Prediction",
    outpath=OUTDIR / "Fig20_rf_species_importance.png"
)

# ROC curves
plot_roc(
    desc_sc, sp_labels, species_order, SPECIES_COLORS,
    "Species Prediction — Random Forest (Physicochemical Descriptors)",
    OUTDIR / "Fig21_rf_species_roc.png"
)

print(f"\n  Classification report (Species):")
for sp in species_order:
    m = rep2.get(sp, {})
    print(f"    {sp:<12}  P={m.get('precision',0):.3f}  "
          f"R={m.get('recall',0):.3f}  F1={m.get('f1-score',0):.3f}  "
          f"n={m.get('support',0):.0f}")

# ── Save predictions CSV ──────────────────────────────────────────────────────
print("\nSaving results CSV ...")
out_csv = OUTDIR / "vdr_rf_results.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as fh:
    cols = ["PyMOL_Ligand_Name","Species","True_Class",
            "Pred_Class","Pred_Species"]
    writer = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
    writer.writeheader()

    # Match class predictions back to full record list
    cls_pred_map = {rec_cls[i]["code"]: pred1[i] for i in range(len(rec_cls))}
    sp_pred_list = list(pred2)

    for i, rec in enumerate(records):
        writer.writerow({
            "PyMOL_Ligand_Name": rec["code"],
            "Species":           rec["species"],
            "True_Class":        rec["cls"],
            "Pred_Class":        cls_pred_map.get(rec["code"], "gemini_excluded"),
            "Pred_Species":      sp_pred_list[i],
        })

print(f"  Saved: {out_csv.name}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RANDOM FOREST COMPLETE")
print("="*55)
print(f"  Model 1 — Ligand class (ECFP6):      {acc1*100:.1f}% accuracy")
print(f"  Model 2 — Species (descriptors):      {acc2*100:.1f}% accuracy")
print("\nFigures saved:")
for i in range(16, 22):
    print(f"  Fig{i}_rf_*.png")
print(f"  vdr_rf_results.csv")
