#!/usr/bin/env python3
"""
Generate molecular formula table for all VDR ligands by class.
Output: vdr_molecular_formulas.csv
"""

import csv
from pathlib import Path
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("conda install -c conda-forge rdkit")

INPUT  = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_final.csv")
OUTPUT = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_molecular_formulas.csv")

BORON_CODES = {"A1MAV","A1MAW","A1MAX","A1MAY","A1MAZ","A1MAU","M7E"}
CLASS_ORDER = ["secosteroid","non_steroidal","steroidal","gemini","boron_cluster"]

with open(INPUT, newline="", encoding="utf-8") as fh:
    sample = fh.read(2000); fh.seek(0)
    delim  = ";" if sample.count(";") > sample.count(",") else ","
    rows   = list(csv.DictReader(fh, delimiter=delim))

results = []

for row in rows:
    code    = row["PyMOL_Ligand_Name"].strip()
    cls     = row["Ligand_Class"].strip()
    species = row["Species"].strip()
    smiles  = row.get("SMILES_stereo","").strip() or row.get("SMILES","").strip()
    name    = row.get("RCSB_Compound_Name","").strip()
    ccd     = row.get("Correct_CCD_Code","").strip() or code
    pdb     = row.get("Protein_PDB","").strip()

    if code in BORON_CODES:
        results.append({
            "Ligand_Class":       cls,
            "PyMOL_Code":         code,
            "Correct_CCD_Code":   ccd,
            "Protein_PDB":        pdb,
            "Species":            species,
            "Molecular_Formula":  "N/A (boron cage)",
            "Exact_MW":           "N/A",
            "SMILES":             smiles[:80],
            "RCSB_Name":          name[:100],
        })
        continue

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol:
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw      = round(Descriptors.ExactMolWt(mol), 4)
    else:
        formula = "parse_failed"
        mw      = ""

    results.append({
        "Ligand_Class":       cls,
        "PyMOL_Code":         code,
        "Correct_CCD_Code":   ccd,
        "Protein_PDB":        pdb,
        "Species":            species,
        "Molecular_Formula":  formula,
        "Exact_MW":           mw,
        "SMILES":             smiles[:80],
        "RCSB_Name":          name[:100],
    })

# Sort by class then code
results.sort(key=lambda x: (CLASS_ORDER.index(x["Ligand_Class"])
                             if x["Ligand_Class"] in CLASS_ORDER else 99,
                             x["PyMOL_Code"]))

# Write CSV
fieldnames = ["Ligand_Class","PyMOL_Code","Correct_CCD_Code","Protein_PDB",
              "Species","Molecular_Formula","Exact_MW","SMILES","RCSB_Name"]

with open(OUTPUT, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(results)

# Print to terminal grouped by class
from collections import Counter
print(f"{'='*80}")
print(f"VDR LIGAND MOLECULAR FORMULAS BY CLASS")
print(f"{'='*80}")

for cls in CLASS_ORDER:
    entries = [r for r in results if r["Ligand_Class"] == cls]
    if not entries:
        continue
    print(f"\n{'='*80}")
    print(f"{cls.upper().replace('_',' ')}  (n={len(entries)})")
    print(f"{'='*80}")
    print(f"  {'Code':<10} {'CCD':<10} {'Formula':<18} {'MW (Da)':<10} {'Name'}")
    print(f"  {'-'*8}   {'-'*8}   {'-'*16}   {'-'*8}   {'-'*40}")
    for r in entries:
        print(f"  {r['PyMOL_Code']:<10} {r['Correct_CCD_Code']:<10} "
              f"{r['Molecular_Formula']:<18} {str(r['Exact_MW']):<10} "
              f"{r['RCSB_Name'][:55]}")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
for cls in CLASS_ORDER:
    entries = [r for r in results if r["Ligand_Class"] == cls]
    if not entries:
        continue
    formulas = [r["Molecular_Formula"] for r in entries
                if r["Molecular_Formula"] not in ("N/A (boron cage)","parse_failed","")]
    # Most common formula
    if formulas:
        most_common = Counter(formulas).most_common(1)[0]
        print(f"  {cls:<20} n={len(entries):3}  most common formula: "
              f"{most_common[0]} (x{most_common[1]})")

print(f"\nOutput saved to: {OUTPUT}")
