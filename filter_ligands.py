#!/usr/bin/env python3
"""
Filter vdr_ligands_full.csv:
- Remove solvents, ions, buffers, crystallographic artifacts
- Keep only true small-molecule VDR ligands
- Output: vdr_ligands_clean.csv
"""

import csv
from pathlib import Path

INPUT  = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_full.csv")
OUTPUT = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_clean.csv")

# All codes to exclude based on Figure 3
EXCLUDE = {
    # Solvents / crystallization agents
    "HOH", "WAT", "DOD", "EDO", "FMT", "ACT", "GOL", "MPD",
    "DMS", "BME", "PEG", "DTT", "HED",
    # Buffer components
    "PO4", "SO4", "MES", "EPE", "IMD",
    # Metal ions
    "MG", "CA", "ZN", "NA", "FE", "CU", "MN", "NI", "CO",
    # Halides and other ions
    "CL", "BR", "IOD", "I",
    # Crystallographic artifacts
    "URL", "UIL", "URV", "UZN", "UIA", "MFH", "OUK",
    # Other biomolecules
    "NAG",
}

with open(INPUT, newline="", encoding="utf-8") as fh:
    sample = fh.read(2000)
    fh.seek(0)
    delim = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(fh, delimiter=delim)
    fieldnames = reader.fieldnames
    all_rows = list(reader)

kept    = []
removed = []

for row in all_rows:
    code = row.get("PyMOL_Ligand_Name", "").strip().upper()
    ltype = row.get("Ligand_Type", "").strip().lower()

    if code in EXCLUDE or ltype in ("ion", "solvent/additive"):
        removed.append(row)
    else:
        kept.append(row)

with open(OUTPUT, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(kept)

print(f"Original rows : {len(all_rows)}")
print(f"Removed       : {len(removed)}")
print(f"Kept          : {len(kept)}")
print(f"\nRemoved codes:")
removed_codes = sorted({r['PyMOL_Ligand_Name'] for r in removed})
for c in removed_codes:
    print(f"  {c}")
print(f"\nOutput → {OUTPUT}")
