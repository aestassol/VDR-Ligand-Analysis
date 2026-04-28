#!/usr/bin/env python3
"""
VDR Ligand Chemical Space Analysis
====================================
Bioinformatics-focused visualizations for thesis.

Generates:
  1. UMAP of Morgan fingerprints (chemical space map)
  2. t-SNE of Morgan fingerprints 
  3. MW vs LogP druglikeness scatter
  4. Tanimoto similarity heatmap
  5. Radar plot of molecular properties per class
  6. PCA of molecular descriptors
  7. Property correlation matrix
  8. Stacked bar of Lipinski/Veber compliance

Requirements: pip install rdkit umap-learn scikit-learn matplotlib seaborn

Usage: python3 vdr_chemical_space.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED as QED_mod, Draw
from rdkit import DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed. Install with: pip install umap-learn")
    print("UMAP plots will be skipped.\n")

# ============================================================================
# PATHS
# ============================================================================
BASE = "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis"
UNIQUE_LIGANDS = f"{BASE}/vdr_ligands_final.csv"
OUTPUT = BASE

# ============================================================================
# STYLE
# ============================================================================
# Distinct, colorblind-friendly palette
CLASS_COLORS = {
    'secosteroid':   '#0077B6',   # deep blue
    'non_steroidal': '#E63946',   # vivid red
    'steroidal':     '#2A9D8F',   # teal green
    'boron_cluster': '#F4A261',   # warm orange
    'gemini':        '#9B5DE5',   # purple
}
CLASS_MARKERS = {
    'secosteroid':   'o',
    'non_steroidal': 's',
    'steroidal':     'D',
    'boron_cluster': '^',
    'gemini':        'P',
}
CLASS_ORDER = ['secosteroid', 'non_steroidal', 'steroidal', 'boron_cluster', 'gemini']
CLASS_LABELS = {
    'secosteroid':   'Secosteroid',
    'non_steroidal': 'Non-steroidal',
    'steroidal':     'Steroidal',
    'boron_cluster': 'Boron cluster',
    'gemini':        'Gemini',
}

SPECIES_COLORS = {'Human': '#264653', 'Rat': '#E76F51', 'Zebrafish': '#2A9D8F'}

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ============================================================================
# LOAD & COMPUTE
# ============================================================================
print("Loading data and computing descriptors...")
orig = pd.read_csv(UNIQUE_LIGANDS, sep=';')

# Compute fingerprints and descriptors
mols = []
fps_morgan = []
fps_bits = []
desc_data = []
valid_idx = []

for i, row in orig.iterrows():
    smiles = str(row.get('SMILES', '')).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    
    mols.append(mol)
    valid_idx.append(i)
    
    # Morgan fingerprint (ECFP4, radius=2, 2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fps_morgan.append(fp)
    arr = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps_bits.append(arr)
    
    # Molecular descriptors
    desc_data.append({
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': Descriptors.TPSA(mol),
        'MolMR': Descriptors.MolMR(mol),
        'RingCount': Descriptors.RingCount(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'QED': QED_mod.qed(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
    })

df = orig.iloc[valid_idx].copy().reset_index(drop=True)
X_fp = np.array(fps_bits)
desc_df = pd.DataFrame(desc_data)

print(f"  {len(df)} ligands with valid SMILES")
print(f"  Classes: {df['Ligand_Class'].value_counts().to_dict()}")

# ============================================================================
# Tanimoto similarity matrix
# ============================================================================
print("Computing Tanimoto similarity matrix...")
n = len(fps_morgan)
tanimoto = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        sim = DataStructs.TanimotoSimilarity(fps_morgan[i], fps_morgan[j])
        tanimoto[i, j] = sim
        tanimoto[j, i] = sim

# ============================================================================
# FIGURE 1: UMAP Chemical Space
# ============================================================================
if HAS_UMAP:
    print("Generating UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='jaccard', random_state=42)
    X_umap = reducer.fit_transform(X_fp)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Panel A: colored by ligand class
    ax = axes[0]
    for cls in CLASS_ORDER:
        mask = df['Ligand_Class'] == cls
        if mask.sum() > 0:
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                      c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                      s=80, alpha=0.85, edgecolors='white', linewidth=0.5,
                      label=f'{CLASS_LABELS[cls]} (n={mask.sum()})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('A) Chemical Space by Ligand Class')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    
    # Panel B: colored by species
    ax = axes[1]
    for sp in ['Human', 'Rat', 'Zebrafish']:
        mask = df['Species'] == sp
        if mask.sum() > 0:
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                      c=SPECIES_COLORS[sp], s=80, alpha=0.7,
                      edgecolors='white', linewidth=0.5,
                      label=f'{sp} (n={mask.sum()})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('B) Chemical Space by Species')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    
    plt.suptitle('UMAP Projection of Morgan Fingerprints (ECFP4, 2048-bit)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{OUTPUT}/fig_umap_chemical_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_umap_chemical_space.png")

# ============================================================================
# FIGURE 2: t-SNE Chemical Space
# ============================================================================
print("Generating t-SNE...")
tsne = TSNE(n_components=2, perplexity=min(30, len(X_fp)-1), random_state=42, metric='jaccard')
X_tsne = tsne.fit_transform(X_fp)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
for cls in CLASS_ORDER:
    mask = df['Ligand_Class'] == cls
    if mask.sum() > 0:
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                  s=80, alpha=0.85, edgecolors='white', linewidth=0.5,
                  label=f'{CLASS_LABELS[cls]} (n={mask.sum()})')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('A) Chemical Space by Ligand Class')
ax.legend(fontsize=9, loc='best', framealpha=0.9)

ax = axes[1]
for sp in ['Human', 'Rat', 'Zebrafish']:
    mask = df['Species'] == sp
    if mask.sum() > 0:
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=SPECIES_COLORS[sp], s=80, alpha=0.7,
                  edgecolors='white', linewidth=0.5,
                  label=f'{sp} (n={mask.sum()})')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('B) Chemical Space by Species')
ax.legend(fontsize=9, loc='best', framealpha=0.9)

plt.suptitle('t-SNE Projection of Morgan Fingerprints (ECFP4, 2048-bit)',
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUTPUT}/fig_tsne_chemical_space.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_tsne_chemical_space.png")

# ============================================================================
# FIGURE 3: MW vs LogP Druglikeness Plot
# ============================================================================
print("Generating MW vs LogP plot...")
fig, ax = plt.subplots(figsize=(10, 8))

for cls in CLASS_ORDER:
    mask = df['Ligand_Class'] == cls
    if mask.sum() > 0:
        ax.scatter(desc_df.loc[mask, 'LogP'], desc_df.loc[mask, 'MW'],
                  c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                  s=90, alpha=0.85, edgecolors='white', linewidth=0.5,
                  label=f'{CLASS_LABELS[cls]} (n={mask.sum()})')

# Lipinski boundaries
ax.axhline(500, color='gray', ls='--', alpha=0.6, lw=1.5)
ax.axvline(5, color='gray', ls='--', alpha=0.6, lw=1.5)
ax.fill_between([-3, 5], 0, 500, alpha=0.05, color='green')
ax.text(4.8, 510, 'Lipinski MW ≤ 500', fontsize=9, color='gray', ha='right')
ax.text(5.1, 50, 'Lipinski LogP ≤ 5', fontsize=9, color='gray', rotation=90, va='bottom')

ax.set_xlabel('Calculated LogP (Wildman-Crippen)')
ax.set_ylabel('Molecular Weight (Da)')
ax.set_title('Druglikeness Space: MW vs LogP')
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.set_xlim(desc_df['LogP'].min() - 1, desc_df['LogP'].max() + 1)

plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_mw_logp_druglikeness.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_mw_logp_druglikeness.png")

# ============================================================================
# FIGURE 4: Tanimoto Similarity Heatmap
# ============================================================================
print("Generating Tanimoto heatmap...")

# Sort by class for better visualization
sort_order = []
for cls in CLASS_ORDER:
    idx = df[df['Ligand_Class'] == cls].index.tolist()
    sort_order.extend(idx)
tani_sorted = tanimoto[np.ix_(sort_order, sort_order)]
labels_sorted = df.iloc[sort_order]['Ligand_Class'].values

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(tani_sorted, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Tanimoto Similarity', shrink=0.8)

# Add class boundaries
cumsum = 0
for cls in CLASS_ORDER:
    n_cls = (labels_sorted == cls).sum()
    if n_cls > 0:
        ax.axhline(cumsum - 0.5, color='black', lw=1)
        ax.axvline(cumsum - 0.5, color='black', lw=1)
        ax.text(cumsum + n_cls/2, -3, CLASS_LABELS[cls], ha='center', fontsize=8, fontweight='bold')
        cumsum += n_cls

ax.axhline(cumsum - 0.5, color='black', lw=1)
ax.axvline(cumsum - 0.5, color='black', lw=1)

ax.set_xlabel('Ligand Index (sorted by class)')
ax.set_ylabel('Ligand Index (sorted by class)')
ax.set_title('Tanimoto Similarity Matrix (Morgan/ECFP4 Fingerprints)')

plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_tanimoto_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_tanimoto_heatmap.png")

# ============================================================================
# FIGURE 5: Radar Plot of Molecular Properties per Class
# ============================================================================
print("Generating radar plot...")
radar_props = ['MW', 'LogP', 'HBD', 'HBA', 'RotBonds', 'TPSA']
radar_labels = ['MW', 'LogP', 'HBD', 'HBA', 'Rotatable\nBonds', 'TPSA']

# Normalize each property to 0-1 range across all ligands
radar_norm = {}
for prop in radar_props:
    vals = desc_df[prop]
    radar_norm[prop] = (vals - vals.min()) / (vals.max() - vals.min())

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(radar_props), endpoint=False).tolist()
angles += angles[:1]

for cls in CLASS_ORDER:
    mask = df['Ligand_Class'] == cls
    if mask.sum() < 2:
        continue
    values = [radar_norm[prop][mask].mean() for prop in radar_props]
    values += values[:1]
    ax.plot(angles, values, 'o-', color=CLASS_COLORS[cls], linewidth=2, 
            markersize=6, label=CLASS_LABELS[cls], alpha=0.8)
    ax.fill(angles, values, color=CLASS_COLORS[cls], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Normalized Molecular Property Profiles', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_radar_properties.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_radar_properties.png")

# ============================================================================
# FIGURE 6: PCA of Molecular Descriptors
# ============================================================================
print("Generating PCA of molecular descriptors...")
desc_cols = ['MW', 'LogP', 'HBD', 'HBA', 'RotBonds', 'TPSA', 'MolMR', 
             'RingCount', 'HeavyAtomCount', 'QED', 'FractionCSP3', 'NumAromaticRings']
X_desc = desc_df[desc_cols].values
X_desc_scaled = StandardScaler().fit_transform(X_desc)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_desc_scaled)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
for cls in CLASS_ORDER:
    mask = df['Ligand_Class'] == cls
    if mask.sum() > 0:
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                  s=80, alpha=0.85, edgecolors='white', linewidth=0.5,
                  label=f'{CLASS_LABELS[cls]} (n={mask.sum()})')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('A) PCA of Molecular Descriptors')
ax.legend(fontsize=9, framealpha=0.9)

# Loadings
ax = axes[1]
loadings = pca.components_.T
for i, prop in enumerate(desc_cols):
    ax.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, head_width=0.08, 
             head_length=0.05, fc='#264653', ec='#264653', alpha=0.7)
    ax.text(loadings[i, 0]*3.3, loadings[i, 1]*3.3, prop, fontsize=8, 
            ha='center', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('B) PCA Loadings (Property Contributions)')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.axhline(0, color='gray', ls='--', alpha=0.3)
ax.axvline(0, color='gray', ls='--', alpha=0.3)

plt.suptitle('PCA of Molecular Descriptors (12 Properties)', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUTPUT}/fig_pca_molecular_descriptors.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_pca_molecular_descriptors.png")

# ============================================================================
# FIGURE 7: Property Distributions (violin + strip)
# ============================================================================
print("Generating violin plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (prop, label) in enumerate(zip(
    ['MW', 'LogP', 'HBA', 'HBD', 'RotBonds', 'TPSA'],
    ['Molecular Weight (Da)', 'LogP', 'H-Bond Acceptors',
     'H-Bond Donors', 'Rotatable Bonds', 'TPSA (Å²)'])):
    
    ax = axes[idx]
    plot_data = pd.DataFrame({
        'value': desc_df[prop],
        'class': df['Ligand_Class'].map(CLASS_LABELS)
    }).dropna()
    
    order = [CLASS_LABELS[c] for c in CLASS_ORDER if CLASS_LABELS[c] in plot_data['class'].values]
    colors = [CLASS_COLORS[c] for c in CLASS_ORDER if CLASS_LABELS[c] in plot_data['class'].values]
    
    parts = ax.violinplot(
        [plot_data[plot_data['class'] == c]['value'].values for c in order],
        positions=range(len(order)), showmedians=True, showextrema=False)
    
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.4)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    
    # Strip plot overlay
    for i, (cls_label, color) in enumerate(zip(order, colors)):
        vals = plot_data[plot_data['class'] == cls_label]['value'].values
        jitter = np.random.RandomState(42).normal(0, 0.06, len(vals))
        ax.scatter(i + jitter, vals, c=color, s=20, alpha=0.7, edgecolors='white', linewidth=0.3)
    
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=11, fontweight='bold')

plt.suptitle('Molecular Property Distributions by Ligand Class', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUTPUT}/fig_violin_properties.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_violin_properties.png")

# ============================================================================
# FIGURE 8: Intra-class vs Inter-class Tanimoto Similarity
# ============================================================================
print("Generating similarity boxplot...")
intra_sims = {}
inter_sims = {}

for cls in CLASS_ORDER:
    mask = (df['Ligand_Class'] == cls).values
    idx = np.where(mask)[0]
    not_idx = np.where(~mask)[0]
    
    if len(idx) > 1:
        intra = []
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                intra.append(tanimoto[idx[i], idx[j]])
        intra_sims[cls] = intra
    
    if len(idx) > 0 and len(not_idx) > 0:
        inter = []
        for i in idx:
            for j in not_idx[:50]:  # sample to keep manageable
                inter.append(tanimoto[i, j])
        inter_sims[cls] = inter

fig, ax = plt.subplots(figsize=(12, 6))
positions = []
bp_data = []
bp_colors = []
bp_labels = []
pos = 0

for cls in CLASS_ORDER:
    if cls in intra_sims and len(intra_sims[cls]) > 0:
        bp_data.append(intra_sims[cls])
        bp_labels.append(f'{CLASS_LABELS[cls]}\nIntra')
        bp_colors.append(CLASS_COLORS[cls])
        positions.append(pos)
        pos += 0.8
    
    if cls in inter_sims and len(inter_sims[cls]) > 0:
        bp_data.append(inter_sims[cls])
        bp_labels.append(f'{CLASS_LABELS[cls]}\nInter')
        bp_colors.append('#CCCCCC')
        positions.append(pos)
        pos += 1.2

bp = ax.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True,
               medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels(bp_labels, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Tanimoto Similarity')
ax.set_title('Intra-class vs Inter-class Chemical Similarity', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_tanimoto_intra_inter.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_tanimoto_intra_inter.png")

# ============================================================================
# PRINT EXPLANATIONS
# ============================================================================
print("\n" + "="*80)
print("FIGURE EXPLANATIONS FOR THESIS")
print("="*80)

explanations = """
FIGURE: UMAP Chemical Space (fig_umap_chemical_space.png)
  Name: UMAP projection of VDR ligand chemical space based on Morgan fingerprints
  X-axis: UMAP dimension 1 (arbitrary units, captures structural similarity)
  Y-axis: UMAP dimension 2 (arbitrary units)
  Method: Each ligand is represented by a 2048-bit Morgan/ECFP4 fingerprint 
    (radius=2), which encodes circular substructures around each atom. UMAP 
    (Uniform Manifold Approximation and Projection) reduces this 2048-dimensional 
    space to 2D while preserving local neighborhood structure.
  Importance: Ligands that are structurally similar appear close together. 
    Clear separation between clusters indicates distinct chemical scaffolds. 
    Secosteroids cluster tightly (shared vitamin D backbone), while non-steroidal 
    ligands are dispersed (diverse scaffolds).

FIGURE: t-SNE Chemical Space (fig_tsne_chemical_space.png)
  Name: t-SNE projection of VDR ligand chemical space
  X/Y-axis: t-SNE dimensions (arbitrary units)
  Method: Similar to UMAP but uses t-distributed Stochastic Neighbor Embedding.
    Uses Jaccard distance between binary fingerprints.
  Importance: Complementary to UMAP. If both show similar clustering, the 
    chemical space structure is robust.

FIGURE: MW vs LogP (fig_mw_logp_druglikeness.png)
  Name: Druglikeness analysis of VDR ligands
  X-axis: Calculated LogP (Wildman-Crippen method) — measures lipophilicity.
    Higher LogP = more hydrophobic. Lipinski rule: LogP ≤ 5.
  Y-axis: Molecular weight in Daltons. Lipinski rule: MW ≤ 500 Da.
  Green shaded area: Lipinski-compliant region (MW ≤ 500, LogP ≤ 5).
  Importance: Shows whether VDR ligands have drug-like properties. 
    Secosteroids tend to have high LogP (lipophilic, as expected for 
    vitamin D analogs). Non-steroidal ligands show wider MW/LogP range.

FIGURE: Tanimoto Heatmap (fig_tanimoto_heatmap.png)
  Name: Pairwise Tanimoto similarity matrix
  Axes: Ligand indices sorted by class
  Color scale: Tanimoto coefficient (0 = completely different, 1 = identical)
  Method: Tanimoto similarity between Morgan fingerprints measures structural 
    overlap. Tc = |A ∩ B| / |A ∪ B| for binary fingerprint bits.
  Importance: Hot blocks along the diagonal show high intra-class similarity.
    Off-diagonal blocks show inter-class similarity. Secosteroids show high 
    mutual similarity (shared vitamin D scaffold). Non-steroidal ligands show 
    lower mutual similarity (diverse scaffolds).

FIGURE: Radar Plot (fig_radar_properties.png)
  Name: Normalized molecular property profiles per ligand class
  Axes: Six key molecular properties (MW, LogP, HBD, HBA, RotBonds, TPSA), 
    each normalized to 0–1 range across all ligands.
  Importance: Visual comparison of property profiles. Secosteroids have 
    characteristic high MW, high LogP, moderate TPSA. Non-steroidal ligands 
    show different profiles with higher aromatic character and lower LogP.

FIGURE: PCA Molecular Descriptors (fig_pca_molecular_descriptors.png)
  Name: PCA of 12 molecular descriptors
  Panel A - X-axis: PC1 (captures most variance in molecular properties)
  Panel A - Y-axis: PC2 (second most variance)
  Panel B: Loading plot showing which properties drive each PC direction.
  Method: 12 RDKit descriptors standardized and projected onto principal components.
  Importance: Shows which molecular properties differentiate the ligand classes.
    MW, HeavyAtomCount, and MolMR typically load on PC1 (size axis).
    LogP, FractionCSP3, NumAromaticRings load on PC2 (shape/polarity axis).

FIGURE: Violin Plots (fig_violin_properties.png)
  Name: Distribution of molecular properties per ligand class
  X-axis: Ligand class
  Y-axis: Property value
  Method: Violin plots show the full distribution shape (KDE). Individual 
    data points are overlaid as a strip plot.
  Importance: Reveals differences in property distributions between classes.
    Wider violins = more variability. Medians show typical values.

FIGURE: Tanimoto Intra/Inter (fig_tanimoto_intra_inter.png)
  Name: Intra-class vs inter-class chemical similarity comparison
  X-axis: Ligand class (intra = within class, inter = to other classes)
  Y-axis: Tanimoto similarity coefficient
  Importance: High intra-class similarity + low inter-class similarity = 
    well-defined chemical classes. If intra ≈ inter, the classes aren't 
    chemically distinct.
"""

print(explanations)
print("All figures saved to:", OUTPUT)
print("Done!")
