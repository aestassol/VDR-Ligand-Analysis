#!/usr/bin/env python3
"""
2D Structure Grid Figures — One per ligand class
Shows key representative ligands with CCD code + shortened IUPAC name

Output:
  Fig25_structures_secosteroid.png
  Fig26_structures_non_steroidal.png
  Fig27_structures_steroidal.png
  Fig28_structures_gemini.png
"""

import csv
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("conda install -c conda-forge rdkit")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.image import imread
    import numpy as np
    from PIL import Image
    import io
except ImportError:
    raise SystemExit("pip install matplotlib pillow")

INPUT  = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_final.csv")
OUTDIR = INPUT.parent

# ── Load data ─────────────────────────────────────────────────────────────────
with open(INPUT, newline="", encoding="utf-8") as fh:
    sample = fh.read(2000); fh.seek(0)
    delim  = ";" if sample.count(";") > sample.count(",") else ","
    rows   = list(csv.DictReader(fh, delimiter=delim))

data = {}
for row in rows:
    code = row["PyMOL_Ligand_Name"].strip()
    data[code] = {
        "ccd":     row.get("Correct_CCD_Code","").strip() or code,
        "name":    row.get("RCSB_Compound_Name","").strip(),
        "cls":     row["Ligand_Class"].strip(),
        "species": row["Species"].strip(),
        "smiles":  row.get("SMILES_stereo","").strip() or row.get("SMILES","").strip(),
    }

# ── Key representatives per class ────────────────────────────────────────────
# Selected based on: scaffold analysis top hits, UMAP cluster representatives,
# RF model key compounds, species diversity

SECOSTEROID_KEY = [
    # Scaffold 1 dominant — classic VDR ligands
    ("VDX",  "Calcitriol (1,25-(OH)₂D₃)"),
    ("MC9",  "Calcipotriol"),
    ("EB1",  "Seocalcitol"),
    # Side chain variants — scaffold 2
    ("VDZ",  "22-oxa analog"),
    ("VD2",  "20-epi analog"),
    # Modified A-ring
    ("C33",  "2α-Propyl analog"),
    ("MVD",  "2α-Methyl analog"),
    ("EIM",  "3-OMe analog"),
    # Unusual side chains
    ("XE4",  "Diyne side chain"),
    ("G72",  "Trifluoro analog"),
    ("ZNE",  "Thiazole side chain"),
    ("ZYD",  "Nor-seco analog"),
]

NON_STEROIDAL_KEY = [
    # Diarylmethane series (scaffold 3)
    ("YR3",  "Diarylmethane (YR3)"),
    ("DS2",  "Diarylmethane (DS2)"),
    ("W07",  "Diaryl ether (W07)"),
    ("O11",  "Diaryl ether (O11)"),
    # Fluorinated diaryl (scaffold 4)
    ("DS5",  "CF₃ diarylmethane"),
    ("6DS",  "CF₃ vinyl analog"),
    # Diarylsilane series (A1L7)
    ("A1L7X","Diarylsilane (CF₃)"),
    ("A1L70","Diarylsilane (unsubst.)"),
    # Other non-steroidal
    ("8VM",  "Thiophene analog"),
    ("484",  "Biphenyl analog"),
    ("LX3",  "Aryl-triene (LX3)"),
    ("A1JD5","Thiazole non-steroidal"),
]

STEROIDAL_KEY = [
    ("4OA",  "Lithocholic acid"),
    ("3KL",  "3-Oxocholic acid"),
    ("LOA",  "3-Acetoxy analog"),
    ("LHP",  "3-Propanoyl analog"),
    ("FKC",  "2-OH-methyl bile acid"),
    ("2U1",  "6-Substituted bile acid"),
    ("2WV",  "6-OH bile acid"),
    ("7SM",  "Biphenyl bile acid"),
]

GEMINI_KEY = [
    ("BIV",  "21-nor-Gemini VDR ligand\n(BIV — bis-side chain)"),
]

# ── Draw function ─────────────────────────────────────────────────────────────
def shorten_name(name, max_len=35):
    """Shorten IUPAC name for display."""
    name = name.replace("~{","").replace("}","").replace("~","")
    if len(name) <= max_len:
        return name
    # Try to break at a comma or bracket
    for sep in [",", " ", "-"]:
        idx = name[:max_len].rfind(sep)
        if idx > max_len//2:
            return name[:idx] + "…"
    return name[:max_len] + "…"


def mol_to_image(smiles, size=(300, 250)):
    """Convert SMILES to PIL Image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().padding = 0.15
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    bio = io.BytesIO(drawer.GetDrawingText())
    return Image.open(bio)


def make_grid_figure(key_list, title, outpath, ncols=4,
                     bg_color="#FAFAFA", header_color="#1565C0"):
    """Create a grid figure of 2D structures."""
    n     = len(key_list)
    nrows = (n + ncols - 1) // ncols
    fw    = ncols * 3.8
    fh    = nrows * 4.2 + 0.8

    fig   = plt.figure(figsize=(fw, fh))
    fig.patch.set_facecolor(bg_color)
    fig.suptitle(title, fontsize=14, fontweight="bold",
                 color=header_color, y=0.98)

    for idx, (code, display_label) in enumerate(key_list):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        ax.set_facecolor("white")
        ax.axis("off")

        rec    = data.get(code, {})
        smiles = rec.get("smiles","")
        ccd    = rec.get("ccd", code)
        name   = rec.get("name","")

        img = mol_to_image(smiles) if smiles else None

        if img:
            ax.imshow(np.array(img))
        else:
            ax.text(0.5, 0.5, "Structure\nunavailable",
                    ha="center", va="center", fontsize=9, color="#999")

        # Label: CCD code bold + display label below
        species_tag = f"[{rec.get('species','')}]"
        label = f"{ccd}  {species_tag}\n{display_label}"
        ax.set_title(label, fontsize=8, fontweight="bold",
                     color="#222222", pad=3,
                     multialignment="center")

        # Thin border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#DDDDDD")
            spine.set_linewidth(0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {outpath.name}")


# ── Generate figures ──────────────────────────────────────────────────────────
print("Generating structure grid figures ...")

make_grid_figure(
    SECOSTEROID_KEY,
    "Key Secosteroid VDR Ligands — Representative Structures",
    OUTDIR / "Fig25_structures_secosteroid.png",
    ncols=4,
    header_color="#1565C0",
)

make_grid_figure(
    NON_STEROIDAL_KEY,
    "Key Non-Steroidal VDR Ligands — Representative Structures",
    OUTDIR / "Fig26_structures_non_steroidal.png",
    ncols=4,
    header_color="#E65100",
)

make_grid_figure(
    STEROIDAL_KEY,
    "Steroidal (Bile Acid) VDR Ligands — All Structures",
    OUTDIR / "Fig27_structures_steroidal.png",
    ncols=4,
    header_color="#2E7D32",
)

make_grid_figure(
    GEMINI_KEY,
    "Gemini VDR Ligand — BIV",
    OUTDIR / "Fig28_structures_gemini.png",
    ncols=1,
    header_color="#6A1B9A",
)

print("\nDone. 4 figures saved:")
for i in range(25, 29):
    print(f"  Fig{i}_structures_*.png")
