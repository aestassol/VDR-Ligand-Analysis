#!/usr/bin/env python3
"""
VDR Ligand Structure Grid — Publication Style
===============================================
Matches manuscript format: colored side chains, property boxes,
numbered labels with class annotations.

Usage: python3 vdr_ligand_structures_pub.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, QED as QED_mod
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE = "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis"
UNIQUE_LIGANDS = f"{BASE}/vdr_ligands_final.csv"
OUTPUT = BASE

# ============================================================================
# SETTINGS
# ============================================================================
MOL_SIZE = (400, 350)       # size per molecule image
COLS = 5                     # molecules per row
FONT_SIZE = 12
HEADER_HEIGHT = 80           # height for property text box above each molecule
BG_COLOR = (255, 255, 255)   # white background

# Colors for highlighting side chain modifications (orange)
HIGHLIGHT_COLOR = (1.0, 0.5, 0.0)  # orange for modified atoms
SECOSTEROID_CORE_SMARTS = "[C]1[C][C]([C]=[C][C]2[C][C][C][C]3([C]2[C][C][C]3)[C])[C][C]1"

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
orig = pd.read_csv(UNIQUE_LIGANDS, sep=';')

# Process all molecules
data = []
for idx, row in orig.iterrows():
    smiles = str(row.get('SMILES', '')).strip()
    if not smiles or smiles == 'nan':
        continue
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    
    AllChem.Compute2DCoords(mol)
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    
    try:
        qed = QED_mod.qed(mol)
    except:
        qed = 0.0
    
    pdb = str(row.get('Protein_PDB', '')).strip()
    ccd = str(row.get('Correct_CCD_Code', row.get('PyMOL_Ligand_Name', ''))).strip()
    species = str(row.get('Species', '')).strip()
    lig_class = str(row.get('Ligand_Class', '')).strip()
    
    data.append({
        'mol': mol,
        'pdb': pdb,
        'ccd': ccd,
        'species': species,
        'class': lig_class,
        'mw': mw,
        'logp': logp,
        'hbd': hbd,
        'hba': hba,
        'rotb': rotb,
        'tpsa': tpsa,
        'qed': qed,
    })

print(f"  {len(data)} valid molecules")

# ============================================================================
# HELPER: Draw single molecule with property header
# ============================================================================
def draw_mol_with_header(entry, idx_num, total_w=400, total_h=430):
    """Draw molecule with property annotation box above it."""
    
    mol = entry['mol']
    
    # Draw molecule using rdMolDraw2D for better control
    drawer = rdMolDraw2D.MolDraw2DSVG(total_w, total_h - HEADER_HEIGHT)
    
    # Set drawing options
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.0
    opts.addAtomIndices = False
    opts.addStereoAnnotation = True
    opts.padding = 0.15
    
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    # Convert SVG to PNG via rdkit
    mol_img = Draw.MolToImage(mol, size=(total_w, total_h - HEADER_HEIGHT))
    
    # Create full image with header
    full_img = Image.new('RGB', (total_w, total_h), BG_COLOR)
    
    # Draw header box
    draw = ImageDraw.Draw(full_img)
    
    # Header background (light blue-gray)
    draw.rectangle([0, 0, total_w, HEADER_HEIGHT], fill=(240, 245, 250), outline=(200, 210, 220))
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE - 2)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE)
            font_bold = font
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE - 2)
        except:
            font = ImageFont.load_default()
            font_bold = font
            font_small = font
    
    # Class abbreviation
    class_abbr = {
        'secosteroid': 'SS', 'non_steroidal': 'NS', 
        'steroidal': 'S', 'boron_cluster': 'BC', 'gemini': 'G'
    }
    cls = class_abbr.get(entry['class'], '?')
    
    # Header text
    title = f"#{idx_num} ({cls})"
    draw.text((8, 4), title, fill=(0, 80, 160), font=font_bold)
    
    # PDB/CCD
    draw.text((8, 20), f"{entry['pdb']}/{entry['ccd']}", fill=(80, 80, 80), font=font_small)
    
    # Properties line 1
    line1 = f"Mw: {entry['mw']:.1f}  LP: {entry['logp']:.2f}  TP: {entry['tpsa']:.1f}"
    draw.text((8, 38), line1, fill=(60, 60, 60), font=font_small)
    
    # Properties line 2
    line2 = f"HBD: {entry['hbd']:.0f}  HBA: {entry['hba']:.0f}  RotB: {entry['rotb']:.0f}"
    draw.text((8, 54), line2, fill=(60, 60, 60), font=font_small)
    
    # Species indicator
    sp_color = {'Human': (38, 70, 83), 'Rat': (231, 111, 81), 'Zebrafish': (42, 157, 143)}
    sp_c = sp_color.get(entry['species'], (100, 100, 100))
    draw.text((total_w - 80, 4), entry['species'][:3], fill=sp_c, font=font_bold)
    
    # Paste molecule image below header
    full_img.paste(mol_img, (0, HEADER_HEIGHT))
    
    return full_img


# ============================================================================
# GENERATE GRIDS PER CLASS
# ============================================================================
CLASS_ORDER = ['secosteroid', 'non_steroidal', 'steroidal', 'gemini']
CLASS_NAMES = {
    'secosteroid': 'Secosteroid',
    'non_steroidal': 'Non-steroidal',
    'steroidal': 'Steroidal',
    'boron_cluster': 'Boron Cluster',
    'gemini': 'Gemini',
}

for cls in CLASS_ORDER:
    cls_data = [d for d in data if d['class'] == cls]
    if not cls_data:
        continue
    
    # Sort by species then PDB
    cls_data.sort(key=lambda x: (x['species'], x['pdb']))
    
    n = len(cls_data)
    cols = min(COLS, n)
    rows = (n + cols - 1) // cols
    
    print(f"\n{CLASS_NAMES[cls]}: {n} ligands ({rows}x{cols} grid)...")
    
    # Generate individual molecule images
    mol_images = []
    for i, entry in enumerate(cls_data):
        img = draw_mol_with_header(entry, i + 1, MOL_SIZE[0], MOL_SIZE[1] + HEADER_HEIGHT)
        mol_images.append(img)
    
    # Assemble grid
    grid_w = cols * MOL_SIZE[0]
    grid_h = rows * (MOL_SIZE[1] + HEADER_HEIGHT)
    
    # Add title bar
    title_h = 50
    grid = Image.new('RGB', (grid_w, grid_h + title_h), BG_COLOR)
    
    # Title
    draw = ImageDraw.Draw(grid)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            title_font = ImageFont.load_default()
    
    title_text = f"{CLASS_NAMES[cls]} Ligands (n={n})"
    draw.text((20, 10), title_text, fill=(30, 40, 56), font=title_font)
    
    # Place molecules
    for i, img in enumerate(mol_images):
        row = i // cols
        col = i % cols
        x = col * MOL_SIZE[0]
        y = title_h + row * (MOL_SIZE[1] + HEADER_HEIGHT)
        grid.paste(img, (x, y))
    
    fname = f"{OUTPUT}/fig_pub_structures_{cls}.png"
    grid.save(fname, dpi=(300, 300))
    print(f"  Saved: fig_pub_structures_{cls}.png")

# ============================================================================
# COMBINED OVERVIEW: Representatives from each class
# ============================================================================
print("\nGenerating combined overview...")

overview_data = []
for cls in CLASS_ORDER:
    cls_data = [d for d in data if d['class'] == cls]
    if not cls_data:
        continue
    
    cls_data.sort(key=lambda x: x['mw'])
    n = len(cls_data)
    
    if n <= 3:
        selected = cls_data
    else:
        indices = np.linspace(0, n - 1, 3, dtype=int)
        selected = [cls_data[i] for i in indices]
    
    overview_data.extend(selected)

n = len(overview_data)
cols = min(4, n)
rows = (n + cols - 1) // cols

mol_images = []
for i, entry in enumerate(overview_data):
    img = draw_mol_with_header(entry, i + 1, MOL_SIZE[0], MOL_SIZE[1] + HEADER_HEIGHT)
    mol_images.append(img)

grid_w = cols * MOL_SIZE[0]
grid_h = rows * (MOL_SIZE[1] + HEADER_HEIGHT) + 50

grid = Image.new('RGB', (grid_w, grid_h), BG_COLOR)
draw = ImageDraw.Draw(grid)
try:
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except:
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        title_font = ImageFont.load_default()

draw.text((20, 10), "VDR Ligand Representatives by Class", fill=(30, 40, 56), font=title_font)

for i, img in enumerate(mol_images):
    row = i // cols
    col = i % cols
    x = col * MOL_SIZE[0]
    y = 50 + row * (MOL_SIZE[1] + HEADER_HEIGHT)
    grid.paste(img, (x, y))

fname = f"{OUTPUT}/fig_pub_structures_overview.png"
grid.save(fname, dpi=(300, 300))
print(f"  Saved: fig_pub_structures_overview.png ({n} molecules)")

# ============================================================================
# PER-SPECIES GRIDS
# ============================================================================
for species in ['Human', 'Rat', 'Zebrafish']:
    sp_data = [d for d in data if d['species'] == species]
    if not sp_data:
        continue
    
    sp_data.sort(key=lambda x: ({'secosteroid': 0, 'non_steroidal': 1, 'steroidal': 2, 
                                   'gemini': 3, 'boron_cluster': 4}.get(x['class'], 5), x['pdb']))
    
    n = len(sp_data)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    
    print(f"\n{species}: {n} ligands...")
    
    mol_images = []
    for i, entry in enumerate(sp_data):
        img = draw_mol_with_header(entry, i + 1, 350, 300 + HEADER_HEIGHT)
        mol_images.append(img)
    
    grid_w = cols * 350
    grid_h = rows * (300 + HEADER_HEIGHT) + 50
    grid = Image.new('RGB', (grid_w, grid_h), BG_COLOR)
    draw = ImageDraw.Draw(grid)
    
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        except:
            title_font = ImageFont.load_default()
    
    draw.text((20, 10), f"{species} VDR Ligands (n={n})", fill=(30, 40, 56), font=title_font)
    
    for i, img in enumerate(mol_images):
        row = i // cols
        col = i % cols
        x = col * 350
        y = 50 + row * (300 + HEADER_HEIGHT)
        grid.paste(img, (x, y))
    
    fname = f"{OUTPUT}/fig_pub_structures_{species.lower()}.png"
    grid.save(fname, dpi=(300, 300))
    print(f"  Saved: fig_pub_structures_{species.lower()}.png")

print("\n" + "="*60)
print("All structure grids generated!")
print(f"Output directory: {OUTPUT}")
