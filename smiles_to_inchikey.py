#!/usr/bin/env python3
"""
1. Re-fetch canonical CACTVS stereo SMILES from RCSB CCD for every
   unique ligand code in vdr_ligands_clean.csv
2. Fall back to existing SMILES if RCSB returns nothing
3. Convert to InChI + InChIKey via RDKit
4. Output: vdr_ligands_inchikey.csv
"""

import csv
import time
from pathlib import Path

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    raise SystemExit("pip install requests")

try:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("conda install -c conda-forge rdkit")

INPUT  = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_clean.csv")
OUTPUT = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/vdr_ligands_inchikey.csv")

RCSB_DELAY = 0.25

# ── HTTP session ──────────────────────────────────────────────────────────────
session = requests.Session()
retry = Retry(total=4, backoff_factor=1.5, status_forcelist=[429,500,502,503,504])
session.mount("https://", HTTPAdapter(max_retries=retry))
session.headers.update({"User-Agent": "VDR-thesis/3.0"})


def fetch_stereo_smiles(code: str) -> str:
    """Fetch canonical CACTVS stereo SMILES from RCSB CCD."""
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{code.upper()}"
    time.sleep(RCSB_DELAY)
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        data = resp.json()
        desc = data.get("rcsb_chem_comp_descriptor") or {}
        smiles = desc.get("smiles_stereo") or desc.get("smiles") or ""
        if not smiles:
            for d in data.get("pdbx_chem_comp_descriptor", []):
                if d.get("type") == "SMILES_CANONICAL":
                    smiles = d.get("descriptor", "")
                    break
        return smiles.strip()
    except Exception:
        return ""


def to_inchikey(smiles: str) -> tuple[str, str, str]:
    """Returns (inchi, inchikey, status)."""
    if not smiles.strip():
        return "", "", "empty_smiles"
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return "", "", "rdkit_parse_failed"
    inchi = MolToInchi(mol)
    if not inchi:
        return "", "", "inchi_failed"
    inchikey = InchiToInchiKey(inchi)
    return inchi, inchikey or "", "ok" if inchikey else "inchikey_failed"


# ── Load CSV ──────────────────────────────────────────────────────────────────
with open(INPUT, newline="", encoding="utf-8") as fh:
    sample = fh.read(2000); fh.seek(0)
    delim = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(fh, delimiter=delim)
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

# ── Re-fetch stereo SMILES ────────────────────────────────────────────────────
unique_codes = sorted({r["PyMOL_Ligand_Name"].strip() for r in rows})
print(f"Fetching stereo SMILES for {len(unique_codes)} unique ligand codes ...")

smiles_map: dict[str, str] = {}
for code in unique_codes:
    stereo = fetch_stereo_smiles(code)
    smiles_map[code] = stereo
    status = "OK" if stereo else "MISSING (will use existing)"
    print(f"  {status}  {code}")

# ── Build output rows ─────────────────────────────────────────────────────────
new_fieldnames = fieldnames + ["SMILES_stereo", "SMILES_source", "InChI", "InChIKey", "InChIKey_status"]
out_rows = []
summary = {"ok": 0, "fallback": 0, "failed": 0}

for row in rows:
    code = row["PyMOL_Ligand_Name"].strip()
    rcsb_smiles     = smiles_map.get(code, "")
    existing_smiles = row.get("SMILES", "").strip()

    if rcsb_smiles:
        final_smiles = rcsb_smiles
        smiles_source = "RCSB_CCD_stereo"
    elif existing_smiles:
        final_smiles = existing_smiles
        smiles_source = "existing_csv_fallback"
        summary["fallback"] += 1
    else:
        final_smiles = ""
        smiles_source = "none"

    inchi, inchikey, status = to_inchikey(final_smiles)

    if status == "ok":
        summary["ok"] += 1
    else:
        summary["failed"] += 1

    out_rows.append({
        **row,
        "SMILES_stereo":   final_smiles,
        "SMILES_source":   smiles_source,
        "InChI":           inchi,
        "InChIKey":        inchikey,
        "InChIKey_status": status,
    })

# ── Write output ──────────────────────────────────────────────────────────────
with open(OUTPUT, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=new_fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\nDone. {len(out_rows)} rows written to {OUTPUT}")
print(f"  InChIKey OK         : {summary['ok']}")
print(f"  Used fallback SMILES: {summary['fallback']}")
print(f"  Failed              : {summary['failed']}")

if summary["failed"]:
    print("\nFailed rows (need manual check):")
    for r in out_rows:
        if r["InChIKey_status"] != "ok":
            print(f"  {r['Protein_PDB']}  {r['PyMOL_Ligand_Name']}  -> {r['InChIKey_status']}")
