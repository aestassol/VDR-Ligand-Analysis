#!/usr/bin/env python3
"""
UMAP: Three fingerprints × two colorings (class + species)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from rdkit import DataStructs
import umap
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis"
orig = pd.read_csv(f"{BASE}/vdr_ligands_final.csv", sep=';')

CLASS_COLORS = {
    'secosteroid': '#0077B6', 'non_steroidal': '#E63946',
    'steroidal': '#2A9D8F', 'boron_cluster': '#F4A261', 'gemini': '#9B5DE5',
}
CLASS_MARKERS = {
    'secosteroid': 'o', 'non_steroidal': 's',
    'steroidal': 'D', 'boron_cluster': '^', 'gemini': 'P',
}
CLASS_ORDER = ['secosteroid', 'non_steroidal', 'steroidal', 'boron_cluster', 'gemini']
CLASS_LABELS = {
    'secosteroid': 'Secosteroid', 'non_steroidal': 'Non-steroidal',
    'steroidal': 'Steroidal', 'boron_cluster': 'Boron cluster', 'gemini': 'Gemini',
}
SPECIES_COLORS = {'Human': '#264653', 'Rat': '#E76F51', 'Zebrafish': '#2A9D8F'}

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# Load and compute fingerprints
print("Computing fingerprints...")
mols, valid_idx = [], []
for i, row in orig.iterrows():
    smiles = str(row.get('SMILES', '')).strip()
    if not smiles or smiles == 'nan':
        continue
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    mols.append(mol)
    valid_idx.append(i)

df = orig.iloc[valid_idx].copy().reset_index(drop=True)
print(f"  {len(df)} valid molecules")

fp_data = {}

# Morgan
bits = []
for mol in mols:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros(2048); DataStructs.ConvertToNumpyArray(fp, arr); bits.append(arr)
fp_data['Morgan (ECFP4)'] = np.array(bits)

# MACCS
bits = []
for mol in mols:
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167); DataStructs.ConvertToNumpyArray(fp, arr); bits.append(arr)
fp_data['MACCS Keys'] = np.array(bits)

# RDKit
bits = []
for mol in mols:
    fp = RDKFingerprint(mol, fpSize=2048)
    arr = np.zeros(2048); DataStructs.ConvertToNumpyArray(fp, arr); bits.append(arr)
fp_data['RDKit Topological'] = np.array(bits)

# ============================================================================
# FIGURE: 2 rows × 3 columns (top = class, bottom = species)
# ============================================================================
print("Generating UMAP projections...")
fig, axes = plt.subplots(2, 3, figsize=(22, 13))

for col_idx, (fp_name, X) in enumerate(fp_data.items()):
    print(f"  UMAP for {fp_name}...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='jaccard', random_state=42)
    X_umap = reducer.fit_transform(X)
    
    # Top row: by class
    ax = axes[0, col_idx]
    for cls in CLASS_ORDER:
        mask = df['Ligand_Class'] == cls
        if mask.sum() > 0:
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                      c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                      s=70, alpha=0.85, edgecolors='white', linewidth=0.5,
                      label=f'{CLASS_LABELS[cls]} (n={mask.sum()})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'{fp_name}')
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    
    # Bottom row: by species
    ax = axes[1, col_idx]
    for sp in ['Human', 'Rat', 'Zebrafish']:
        mask = df['Species'] == sp
        if mask.sum() > 0:
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                      c=SPECIES_COLORS[sp], s=70, alpha=0.7,
                      edgecolors='white', linewidth=0.5,
                      label=f'{sp} (n={mask.sum()})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'{fp_name}')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)

# Row labels
axes[0, 0].annotate('By Ligand Class', xy=(-0.25, 0.5), xycoords='axes fraction',
                     fontsize=16, fontweight='bold', rotation=90, va='center', ha='center')
axes[1, 0].annotate('By Species', xy=(-0.25, 0.5), xycoords='axes fraction',
                     fontsize=16, fontweight='bold', rotation=90, va='center', ha='center')

plt.suptitle('UMAP Chemical Space: Three Fingerprint Representations',
             fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0.03, 0, 1, 0.96])
plt.savefig(f'{BASE}/fig_umap_class_species_3fp.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_umap_class_species_3fp.png")
print("Done!")
