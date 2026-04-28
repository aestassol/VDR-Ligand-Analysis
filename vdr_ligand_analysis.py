#!/usr/bin/env python3
"""
VDR Ligand Molecular Property Analysis
=======================================
Generates publication-quality figures for ligand property distributions:
  - KDE distributions per species and ligand class
  - Boxplots comparing species
  - Lipinski/Veber rule-of-5 analysis
  - All ligands vs unique ligands comparison

Usage: python3 vdr_ligand_analysis.py

Edit PATHS below to match your setup.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS - edit these
# ============================================================================
# Unique ligands (160 entries, one per unique ligand)
UNIQUE_LIGANDS = "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_final.csv"
# All ligands (265 entries, one per PDB structure)
ALL_LIGANDS = "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_full.csv"
# Verification file (CCD corrections)
VERIFICATION = "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_verification.csv"
# Output directory
OUTPUT_DIR = "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/"

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
orig = pd.read_csv(UNIQUE_LIGANDS, sep=';')
full = pd.read_csv(ALL_LIGANDS, sep=';')

# Build classification for ALL ligands (not just unique)
# Map CCD codes to classes from verified unique file
ccd_to_class = dict(zip(orig['Correct_CCD_Code'].str.strip(), orig['Ligand_Class'].str.strip()))

# Apply CCD corrections from verification file
try:
    vf = pd.read_csv(VERIFICATION, sep=';')
    ccd_corrections = {}
    for _, row in vf.iterrows():
        old = str(row['PyMOL_Code_in_file']).strip()
        new = str(row['RCSB_Correct_Code']).strip()
        if old != new:
            ccd_corrections[old] = new
except:
    ccd_corrections = {}

# Add manually confirmed classifications for ligands not in unique file
extra_class = {
    '23R': 'secosteroid', '3EV': 'secosteroid', 'ED9': 'secosteroid',
    '41W': 'secosteroid', 'VD5': 'secosteroid', 'VDB': 'secosteroid',
    'JC1': 'secosteroid', 'YI2': 'secosteroid', 'TKD': 'secosteroid',
    'YA2': 'secosteroid', 'YS9': 'secosteroid', 'YSV': 'secosteroid',
    '8J3': 'secosteroid', 'H97': 'secosteroid', '9CW': 'secosteroid',
    '9CZ': 'secosteroid', 'UIA': 'secosteroid',
    'EJO': 'non_steroidal', 'FKF': 'steroidal',
}
ccd_to_class.update(extra_class)

# Classify all ligands in the full file
small_mol = full[full['Ligand_Type'] == 'small-molecule ligand'].copy()
small_mol['correct_ccd'] = small_mol['PyMOL_Ligand_Name'].str.strip().map(
    lambda x: ccd_corrections.get(x, x))
small_mol['Ligand_Class'] = small_mol['correct_ccd'].map(ccd_to_class)

# Compute molecular descriptors using RDKit
print("Computing molecular descriptors...")
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as QED_module
    
    mol_data = []
    for _, row in small_mol.iterrows():
        smiles = str(row.get('SMILES', '')).strip().split('\n')[0]
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol:
            mol_data.append({
                'MW': Descriptors.MolWt(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'LogP': Descriptors.MolLogP(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'TPSA': Descriptors.TPSA(mol),
                'MolMR': Descriptors.MolMR(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
                'QED': QED_module.qed(mol),
                'RingCount': Descriptors.RingCount(mol),
                'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            })
        else:
            mol_data.append({})
    
    mol_df = pd.DataFrame(mol_data)
    for col in mol_df.columns:
        small_mol[col] = mol_df[col].values
    
    HAS_RDKIT = True
    print(f"  RDKit descriptors computed for {mol_df['MW'].notna().sum()}/{len(small_mol)} ligands")
except ImportError:
    HAS_RDKIT = False
    print("  RDKit not available, using pre-computed MW/HBD/HBA/LogP/RotBonds/TPSA from file")
    # Use values from original file where available
    for col in ['MW', 'HBD', 'HBA', 'LogP', 'RotBonds', 'TPSA']:
        if col not in small_mol.columns or small_mol[col].isna().all():
            pdb_to_val = dict(zip(orig['Protein_PDB'].str.upper(), orig[col]))
            small_mol[col] = small_mol['Protein_PDB'].str.upper().map(pdb_to_val)

# Binary classification
small_mol['ligand_binary'] = small_mol['Ligand_Class'].map({
    'secosteroid': 'S', 'steroidal': 'S', 'gemini': 'S',
    'non_steroidal': 'NS', 'boron_cluster': 'NS',
})

# Also prepare unique ligands dataset
unique = orig.copy()
unique['ligand_binary'] = unique['Ligand_Class'].map({
    'secosteroid': 'S', 'steroidal': 'S', 'gemini': 'S',
    'non_steroidal': 'NS', 'boron_cluster': 'NS',
})

print(f"\nAll ligands: {len(small_mol)} (classified: {small_mol['Ligand_Class'].notna().sum()})")
print(f"Unique ligands: {len(unique)}")
print(f"\nAll ligands per species:")
for sp in ['Human', 'Rat', 'Zebrafish']:
    sub = small_mol[small_mol['Species'] == sp]
    print(f"  {sp}: {sub['Ligand_Class'].value_counts().to_dict()}")

# ============================================================================
# FIGURE 1: KDE Distributions of Key Properties (All Ligands, by Species)
# ============================================================================
print("\nGenerating Figure 1: KDE by species...")
properties = ['MW', 'LogP', 'HBA', 'HBD', 'RotBonds', 'TPSA']
prop_labels = ['Molecular Weight (Da)', 'LogP', 'H-Bond Acceptors', 
               'H-Bond Donors', 'Rotatable Bonds', 'TPSA (Å²)']
sp_colors = {'Human': '#2E86AB', 'Rat': '#E84855', 'Zebrafish': '#27AE60'}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (prop, label) in enumerate(zip(properties, prop_labels)):
    ax = axes[idx]
    for sp in ['Human', 'Rat', 'Zebrafish']:
        data = small_mol[(small_mol['Species'] == sp) & small_mol[prop].notna()][prop].values
        if len(data) > 2 and np.std(data) > 0:
            kde = gaussian_kde(data, bw_method=0.3)
            x_range = np.linspace(data.min() - data.std(), data.max() + data.std(), 200)
            ax.fill_between(x_range, kde(x_range), alpha=0.3, color=sp_colors[sp])
            ax.plot(x_range, kde(x_range), color=sp_colors[sp], linewidth=2, label=f'{sp} (n={len(data)})')
    
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_title(label, fontsize=12, fontweight='bold')

plt.suptitle('VDR Ligand Property Distributions by Species (All Ligands)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR + 'fig_ligand_kde_species.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_ligand_kde_species.png")

# ============================================================================
# FIGURE 2: KDE Distributions by Ligand Class (NS vs S)
# ============================================================================
print("Generating Figure 2: KDE by ligand class...")
class_colors = {'S': '#2E86AB', 'NS': '#E84855'}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (prop, label) in enumerate(zip(properties, prop_labels)):
    ax = axes[idx]
    for lt in ['S', 'NS']:
        data = small_mol[(small_mol['ligand_binary'] == lt) & small_mol[prop].notna()][prop].values
        if len(data) > 2 and np.std(data) > 0:
            kde = gaussian_kde(data, bw_method=0.3)
            x_range = np.linspace(data.min() - data.std(), data.max() + data.std(), 200)
            ax.fill_between(x_range, kde(x_range), alpha=0.3, color=class_colors[lt])
            ax.plot(x_range, kde(x_range), color=class_colors[lt], linewidth=2, 
                   label=f'{lt} (n={len(data)})')
    
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title(label, fontsize=12, fontweight='bold')

plt.suptitle('VDR Ligand Property Distributions: S/SS vs NS (All Ligands)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR + 'fig_ligand_kde_class.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_ligand_kde_class.png")

# ============================================================================
# FIGURE 3: Boxplots by Species (All vs Unique)
# ============================================================================
print("Generating Figure 3: Boxplots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, (prop, label) in enumerate(zip(properties, prop_labels)):
    ax = axes_flat[idx]
    
    # Prepare data for boxplot
    bp_data = []
    bp_labels = []
    bp_colors = []
    
    for sp in ['Human', 'Rat', 'Zebrafish']:
        # All ligands
        all_data = small_mol[(small_mol['Species'] == sp) & small_mol[prop].notna()][prop].values
        # Unique ligands
        uni_data = unique[(unique['Species'] == sp) & unique[prop].notna()][prop].values
        
        bp_data.append(all_data)
        bp_data.append(uni_data)
        bp_labels.extend([f'{sp}\nAll (n={len(all_data)})', f'{sp}\nUniq (n={len(uni_data)})'])
        bp_colors.extend([sp_colors[sp], sp_colors[sp]])
    
    bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2))
    
    for i, (patch, color) in enumerate(zip(bp['boxes'], bp_colors)):
        alpha = 0.8 if i % 2 == 0 else 0.4  # lighter for unique
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
    
    ax.set_ylabel(label, fontsize=10)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_title(label, fontsize=11, fontweight='bold')

plt.suptitle('VDR Ligand Properties: All vs Unique Ligands', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR + 'fig_ligand_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_ligand_boxplots.png")

# ============================================================================
# FIGURE 4: Detailed Class Comparison (5 classes) 
# ============================================================================
print("Generating Figure 4: Detailed class comparison...")
class_order = ['secosteroid', 'steroidal', 'gemini', 'non_steroidal', 'boron_cluster']
class_colors_detail = {
    'secosteroid': '#2E86AB', 'steroidal': '#1B4965', 'gemini': '#5FA8D3',
    'non_steroidal': '#E84855', 'boron_cluster': '#FF6B6B',
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, (prop, label) in enumerate(zip(properties, prop_labels)):
    ax = axes_flat[idx]
    
    bp_data = []
    bp_labels_list = []
    colors = []
    
    for cls in class_order:
        data = small_mol[(small_mol['Ligand_Class'] == cls) & small_mol[prop].notna()][prop].values
        if len(data) > 0:
            bp_data.append(data)
            bp_labels_list.append(f'{cls}\n(n={len(data)})')
            colors.append(class_colors_detail[cls])
    
    if bp_data:
        bp = ax.boxplot(bp_data, labels=bp_labels_list, patch_artist=True, widths=0.6,
                       medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
    
    ax.set_ylabel(label, fontsize=10)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_title(label, fontsize=11, fontweight='bold')

plt.suptitle('VDR Ligand Properties by Detailed Class (All Ligands)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR + 'fig_ligand_class_detail.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_ligand_class_detail.png")

# ============================================================================
# FIGURE 5: Lipinski/Veber Rule Compliance
# ============================================================================
print("Generating Figure 5: Lipinski/Veber compliance...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (dataset, title, data_source) in enumerate([
    ('All Ligands', 'All Ligands', small_mol),
    ('Unique Ligands', 'Unique Ligands', unique),
    ('By Class', 'By Ligand Class (All)', small_mol),
]):
    ax = axes[idx]
    
    if idx < 2:
        # By species
        categories = ['Human', 'Rat', 'Zebrafish']
        lipinski_pass = []
        veber_pass = []
        drug_pass = []
        
        for sp in categories:
            sub = data_source[data_source['Species'] == sp]
            n = len(sub)
            if n == 0: continue
            
            # Lipinski: MW<=500, LogP<=5, HBD<=5, HBA<=10
            lip = sub[(sub['MW'] <= 500) & (sub['LogP'] <= 5) & 
                      (sub['HBD'] <= 5) & (sub['HBA'] <= 10)]
            lipinski_pass.append(len(lip) / n * 100)
            
            # Veber: RotBonds<=10, TPSA<=140
            veb = sub[(sub['RotBonds'] <= 10) & (sub['TPSA'] <= 140)]
            veber_pass.append(len(veb) / n * 100)
            
            # Both
            both = sub[(sub['MW'] <= 500) & (sub['LogP'] <= 5) & 
                       (sub['HBD'] <= 5) & (sub['HBA'] <= 10) &
                       (sub['RotBonds'] <= 10) & (sub['TPSA'] <= 140)]
            drug_pass.append(len(both) / n * 100)
        
        x = np.arange(len(categories))
        w = 0.25
        ax.bar(x - w, lipinski_pass, w, label='Lipinski', color='#2E86AB', alpha=0.8)
        ax.bar(x, veber_pass, w, label='Veber', color='#27AE60', alpha=0.8)
        ax.bar(x + w, drug_pass, w, label='Both', color='#E84855', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        
    else:
        # By ligand class
        categories = ['secosteroid', 'non_steroidal', 'steroidal']
        lipinski_pass = []
        veber_pass = []
        drug_pass = []
        
        for cls in categories:
            sub = data_source[data_source['Ligand_Class'] == cls]
            n = max(len(sub), 1)
            lip = sub[(sub['MW'] <= 500) & (sub['LogP'] <= 5) & 
                      (sub['HBD'] <= 5) & (sub['HBA'] <= 10)]
            lipinski_pass.append(len(lip) / n * 100)
            veb = sub[(sub['RotBonds'] <= 10) & (sub['TPSA'] <= 140)]
            veber_pass.append(len(veb) / n * 100)
            both = sub[(sub['MW'] <= 500) & (sub['LogP'] <= 5) & 
                       (sub['HBD'] <= 5) & (sub['HBA'] <= 10) &
                       (sub['RotBonds'] <= 10) & (sub['TPSA'] <= 140)]
            drug_pass.append(len(both) / n * 100)
        
        x = np.arange(len(categories))
        w = 0.25
        ax.bar(x - w, lipinski_pass, w, label='Lipinski', color='#2E86AB', alpha=0.8)
        ax.bar(x, veber_pass, w, label='Veber', color='#27AE60', alpha=0.8)
        ax.bar(x + w, drug_pass, w, label='Both', color='#E84855', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([c[:12] for c in categories], fontsize=9)
    
    ax.set_ylim(0, 110)
    ax.set_ylabel('Pass Rate (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(100, color='gray', ls='--', alpha=0.3)

plt.suptitle('Lipinski & Veber Rule Compliance', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR + 'fig_ligand_lipinski_veber.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_ligand_lipinski_veber.png")

# ============================================================================
# FIGURE 6: MW Distribution Histogram (classic KDa plot)
# ============================================================================
print("Generating Figure 6: MW histogram...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# All ligands
ax = axes[0]
for sp in ['Human', 'Rat', 'Zebrafish']:
    data = small_mol[(small_mol['Species'] == sp) & small_mol['MW'].notna()]['MW'].values
    ax.hist(data, bins=20, alpha=0.5, color=sp_colors[sp], label=f'{sp} (n={len(data)})',
            edgecolor='white', linewidth=0.5)
ax.axvline(500, color='red', ls='--', lw=2, alpha=0.7, label='Lipinski MW limit (500)')
ax.set_xlabel('Molecular Weight (Da)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('All Ligands', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)

# Unique ligands
ax = axes[1]
for sp in ['Human', 'Rat', 'Zebrafish']:
    data = unique[(unique['Species'] == sp) & unique['MW'].notna()]['MW'].values
    ax.hist(data, bins=20, alpha=0.5, color=sp_colors[sp], label=f'{sp} (n={len(data)})',
            edgecolor='white', linewidth=0.5)
ax.axvline(500, color='red', ls='--', lw=2, alpha=0.7, label='Lipinski MW limit (500)')
ax.set_xlabel('Molecular Weight (Da)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Unique Ligands', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)

plt.suptitle('Molecular Weight Distribution', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR + 'fig_ligand_mw_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_ligand_mw_histogram.png")

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for dataset_name, data_source in [('ALL LIGANDS', small_mol), ('UNIQUE LIGANDS', unique)]:
    print(f"\n{dataset_name}:")
    print(f"{'Species':<12} {'N':>4} {'MW':>12} {'LogP':>12} {'HBD':>8} {'HBA':>8} {'RotBonds':>10} {'TPSA':>12}")
    print("-"*80)
    
    for sp in ['Human', 'Rat', 'Zebrafish', 'All']:
        if sp == 'All':
            sub = data_source
        else:
            sub = data_source[data_source['Species'] == sp]
        
        n = len(sub)
        stats = []
        for prop in ['MW', 'LogP', 'HBD', 'HBA', 'RotBonds', 'TPSA']:
            vals = sub[prop].dropna()
            if len(vals) > 0:
                stats.append(f'{vals.mean():.1f}±{vals.std():.1f}')
            else:
                stats.append('N/A')
        
        print(f"{sp:<12} {n:>4} {stats[0]:>12} {stats[1]:>12} {stats[2]:>8} {stats[3]:>8} {stats[4]:>10} {stats[5]:>12}")

print("\n\nAll figures saved to:", OUTPUT_DIR)
print("Done!")
