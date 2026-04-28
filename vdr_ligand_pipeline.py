#!/usr/bin/env python3
"""
VDR Ligand Pipeline — Full Rebuild
====================================
Thesis: Non-steroidal and steroidal ligands selectively induce structural
changes in vitamin D receptor.
Author: Dana · Nazarbayev University BIOL 490/491
Supervisor: Dr. Ferdinand Molnár

Pipeline:
  1. Scan All_Ligands analysis/ for PDB files
  2. Match filenames to protein IDs from species lists
  3. Map each protein → PyMOL ligand code(s)
  4. Resolve each unique ligand code → SMILES via RCSB CCD only
  5. Classify ligand type (small-molecule / ion / solvent / cofactor / unknown)
  6. Output structured CSV with all metadata

Output columns:
  Species | Protein_PDB | Ligand_File | PyMOL_Ligand_Name | Ligand_Type |
  Proposed_Identity | SMILES | Confidence | Notes
"""

import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    sys.exit("ERROR: pip install requests")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(
    "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis"
)
OUTPUT_CSV  = Path(
    "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/"
    "vdr_ligands_full.csv"
)
LOG_FILE = Path(
    "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/"
    "pipeline_full.log"
)

RCSB_DELAY = 0.25   # seconds between API calls

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Species lists ─────────────────────────────────────────────────────────────
SPECIES = {
    "Human": {
        "1IE8","2HAM","2HAR","2HAS","2HB7","2HB8","3A2I","3A2J","3A3Z","3A40",
        "3AUQ","3AUR","3AX8","3AZ1","3AZ2","3AZ3","3B0T","3CS4","3CS6","3KPZ",
        "3M7R","3OGT","3TKC","3W0A","3W0C","3W0Y","3WGP","3WWR","3X31","3X36",
        "4G2I","5GT4","5V39","8IQN","8IQT","1DB1","1IE9","1S0Z","1S19","1TXI",
        "3A78","3P8X","3VHW","4ITE","4ITF","5YSY","5YT2","7QPP",
    },
    "Rat": {
        "2ZL9","2ZLA","3VT4","3VT5","3VT6","3VT8","3VT9","3VTB","3VTC","3VTD",
        "2ZLC","3VT3","2ZMH","2ZMJ","2ZXM","2ZXN","3AFR","2ZMI","5H1E","5XPL",
        "3VT7","1RJK","1RK3","1RKG","1RKH","2O4J","2O4R","2ZFX","3A2H","3AUN",
        "3VJS","3VJT","3VRT","3VRU","3VRV","3VRW","3W0G","3W0H","3W0I","3W0J",
        "3W5P","3W5Q","3W5R","3W5T","3WT5","3WT6","3WT7","3WTQ","4YNK","5AWJ",
        "5AWK","5B41","5B5B","5GIC","5GID","5GIE","5XPM","5XPN","5XPO","5XPP",
        "5XUQ","5XZF","5XZH","5ZWE","5ZWF","5ZWH","5ZWI","6JEZ","6K5O","7C7V",
        "7C7W","7VQP","9M10","9M11","9M12","9M13","9M14","9M15","9M16","9M17",
        "9M18","9M19","9M1A","9M1B","9M1C","9M1D","9VOG","9VOH","9VOI","9VOJ",
        "9VOK","9VOL",
    },
    "Zebrafish": {
        "6XZH","6XZI","6XZV","6XZJ","6XZK","3O1D","3O1E","4FHH","4FHI","4IA1",
        "4IA2","4IA3","4IA7","4Q0A","4RUO","5LGA","8P9X","8PWD","8PWE","8PWF",
        "8PWM","8PZ6","8PZ7","8PZ8","8PZ9","8PZB","9EZ1","9EZ2","9FW8","9GY8",
        "9GYA","9GYC","9GYJ","9GYK","9RCG","2HBH","2HC4","2HCD","3DR1","4G1D",
        "4G1Y","4G1Z","4G20","4G21","4G2H","4RUJ","4RUP","5E7V","5MX7","5NKY",
        "5NMA","5NMB","5OW7","5OW9","5OWD","6FO7","6FO8","6FO9","6FOB","6FOD",
        "6T2M","7BNS","7BNU","7BO6","7OXZ","7OY4","7ZFG","7ZFX","8CK5","8CKC",
        "8P9W","9EYR","9FBF","7B39","7OXU",
    },
    "Sea_Lamprey": {
        "7QPI",
    },
}

# ── PyMOL ligand mapping ──────────────────────────────────────────────────────
# protein_id → list of ligand codes (preserving order / multiplicity)
PYMOL_MAP = {
    "1DB1":  ["VDX"],
    "1IE8":  ["KH1"],
    "1IE9":  ["VDX"],
    "1RJK":  ["VDZ"],
    "1RK3":  ["VDX"],
    "1RKG":  ["VD1"],
    "1RKH":  ["VD2"],
    "1S0Z":  ["EB1"],
    "1S19":  ["MC9"],
    "1TXI":  ["TX5"],
    "2HAM":  ["C33"],
    "2HAR":  ["OCC"],
    "2HAS":  ["C20"],
    "2HB7":  ["O1C"],
    "2HB8":  ["MVD"],
    "2HBH":  ["XE4"],
    "2HC4":  ["VDX"],
    "2HCD":  ["BIV"],
    "2O4J":  ["VD4"],
    "2O4R":  ["VD5"],
    "2ZFX":  ["YR3"],
    "2ZL9":  ["VDA"],
    "2ZLA":  ["VDB"],
    "2ZLC":  ["VDX"],
    "2ZMH":  ["NYA"],
    "2ZMI":  ["TT2","EDO","FMT"],
    "2ZMJ":  ["MI4"],
    "2ZXM":  ["JB1"],
    "2ZXN":  ["JC1"],
    "3A2H":  ["TEJ"],
    "3A2I":  ["TEJ"],
    "3A2J":  ["TEJ"],
    "3A3Z":  ["2MV"],
    "3A40":  ["23R"],
    "3A78":  ["3EV"],
    "3AFR":  ["ICJ"],
    "3AUN":  ["YR4"],
    "3AUQ":  ["CA9"],
    "3AUR":  ["CA9"],
    "3AX8":  ["EIM"],
    "3AZ1":  ["DS2"],
    "3AZ2":  ["DS3"],
    "3AZ3":  ["DS6"],
    "3B0T":  ["MCZ"],
    "3CS4":  ["COV"],
    "3CS6":  ["OCO"],
    "3DR1":  ["CSD"],
    "3KPZ":  ["ZNE"],
    "3M7R":  ["VDX"],
    "3O1D":  ["G72"],
    "3O1E":  ["H97"],
    "3OGT":  ["FMV"],
    "3P8X":  ["ZYD"],
    "3TKC":  ["FMV"],
    "3VHW":  ["VHW"],
    "3VJS":  ["1QS"],
    "3VJT":  ["1QR"],
    "3VRT":  ["YS2"],
    "3VRU":  ["YS3"],
    "3VRV":  ["YSD"],
    "3VRW":  ["YS5"],
    "3VT3":  ["VDX","EDO","FMT","FMT"],
    "3VT4":  ["5YI"],
    "3VT5":  ["YI2"],
    "3VT6":  ["5YI"],
    "3VT7":  ["VDX"],
    "3VT8":  ["YI3"],
    "3VT9":  ["YI4"],
    "3VTB":  ["TKA"],
    "3VTC":  ["TK3","EDO"],
    "3VTD":  ["TKD"],
    "3W0A":  ["DS5"],
    "3W0C":  ["6DS"],
    "3W0G":  ["W07"],
    "3W0H":  ["W12"],
    "3W0I":  ["O11"],
    "3W0J":  ["TQ8"],
    "3W0Y":  ["DS4"],
    "3W5P":  ["4OA"],
    "3W5Q":  ["3KL"],
    "3W5R":  ["LOA"],
    "3W5T":  ["LHP"],
    "3WGP":  ["ED9"],
    "3WT5":  ["YA1"],
    "3WT6":  ["YA1"],
    "3WT7":  ["YA2"],
    "3WTQ":  ["YS9"],
    "3WWR":  ["3AJ"],
    "3X31":  ["41V"],
    "3X36":  ["41W"],
    "4FHH":  ["OU3"],
    "4FHI":  ["OS4"],
    "4G1D":  ["OVK"],
    "4G1Y":  ["OVO"],
    "4G1Z":  ["OVP"],
    "4G20":  ["464"],
    "4G21":  ["OVP"],
    "4G2H":  ["OVQ"],
    "4G2I":  ["OVQ"],
    "4IA1":  ["BIV"],
    "4IA2":  ["BIV"],
    "4IA3":  ["BIV"],
    "4IA7":  ["BIV"],
    "4ITE":  ["TEY"],
    "4ITF":  ["TFY"],
    "4Q0A":  ["4OA","4OA"],
    "4RUJ":  ["VDX"],
    "4RUO":  ["BIV"],
    "4RUP":  ["H97"],
    "4YNK":  ["YW2"],
    "5AWJ":  ["YSL"],
    "5AWK":  ["YSE"],
    "5B41":  ["YSV"],
    "5B5B":  ["AKX","AKX"],
    "5E7V":  ["M7E"],
    "5GIC":  ["DLC"],
    "5GID":  ["VDP"],
    "5GIE":  ["VDP","VDP"],
    "5GT4":  ["2KB"],
    "5H1E":  ["VDX"],
    "5LGA":  ["6VH"],
    "5MX7":  ["D3V"],
    "5NKY":  ["91W"],
    "5NMA":  ["9CW"],
    "5NMB":  ["9CZ"],
    "5OW7":  ["AYK"],
    "5OW9":  ["AYT"],
    "5OWD":  ["BOE"],
    "5V39":  ["8VM"],
    "5XPL":  ["8C9"],
    "5XPM":  ["8CO"],
    "5XPN":  ["8BF"],
    "5XPO":  ["8FF"],
    "5XZF":  ["8J0","FMT"],
    "5XZH":  ["8J3"],
    "5YSY":  ["90L"],
    "5YT2":  ["900"],
    "5ZWE":  ["9KO"],
    "5ZWF":  ["9KR"],
    "5ZWH":  ["9KO","9N9"],
    "5ZWI":  ["9KX"],
    "6FO7":  ["LX3"],
    "6FO8":  ["DZT"],
    "6FO9":  ["E0E"],
    "6FOB":  ["DZW"],
    "6FOD":  ["E09"],
    "6JEZ":  ["YSV","EJ0"],
    "6K5O":  ["DOO"],
    "6T2M":  ["M9Q"],
    "6XZH":  ["VDX"],
    "6XZI":  ["VDX","ACT"],
    "6XZJ":  ["VDX","ACT"],
    "6XZK":  ["VDX","ACT","ACT","UIA"],
    "6XZV":  ["VDX","UIA"],
    "7B39":  ["TOH","ACT"],
    "7BNS":  ["U5W"],
    "7BNU":  ["U5W"],
    "7BO6":  ["FKC"],
    "7C7V":  ["FKI","FMT"],
    "7C7W":  ["FKF","FMT"],
    "7OXU":  ["2Q1"],
    "7OXZ":  ["2U1"],
    "7OY4":  ["2WV"],
    "7QPI":  ["VDX"],
    "7QPP":  ["VDX"],
    "7VQP":  ["7SM"],
    "7ZFG":  ["ACT","IV5"],
    "7ZFX":  ["ACT","IV1"],
    "8CK5":  ["ACT","LY0"],
    "8CKC":  ["ACT","LYU"],
    "8IQN":  ["SRF"],
    "8IQT":  ["SVQ"],
    "8P9W":  ["XD0"],
    "8P9X":  ["ACT","FQ6"],
    "8PWD":  ["FT9"],
    "8PWE":  ["ACT","FUW"],
    "8PWF":  ["FVE"],
    "8PWM":  ["ACT","HXI"],
    "8PZ6":  ["IGS"],
    "8PZ7":  ["IGO"],
    "8PZ8":  ["IFU"],
    "8PZ9":  ["IFK"],
    "8PZB":  ["I72"],
    "9EYR":  ["A1H"],
    "9EZ1":  ["A1H","ACT"],
    "9EZ2":  ["A1H","ACT"],
    "9FBF":  ["A11B"],
    "9FW8":  ["A11G"],
    "9GY8":  ["A11Q"],
    "9GYA":  ["ACT","A11Q"],
    "9GYC":  ["ACT","A11Q"],
    "9GYJ":  ["ACT","A11Q"],
    "9GYK":  ["ACT","A11Q"],
    "9M10":  ["A1L7"],
    "9M11":  ["A1L7"],
    "9M12":  ["A1L7","A1L7"],
    "9M13":  ["A1L7"],
    "9M14":  ["A1L7"],
    "9M15":  ["A1L7"],
    "9M16":  ["A1L7"],
    "9M17":  ["A1L7"],
    "9M18":  ["A1L7"],
    "9M19":  ["A1L7"],
    "9M1A":  ["A1L7"],
    "9M1B":  ["A1L7"],
    "9M1C":  ["A1L8"],
    "9M1D":  ["A1L8"],
    "9RCG":  ["A1ID"],
    "9VOG":  ["A1MA"],
    "9VOH":  ["A1MA"],
    "9VOI":  ["A1MA"],
    "9VOJ":  ["A1MA"],
    "9VOK":  ["A1MA"],
    "9VOL":  ["A1MA"],
}

# ── Ligand type classification ────────────────────────────────────────────────
# Maps ligand code → (ligand_type, proposed_identity)
LIGAND_TYPE_MAP = {
    # Solvents / additives / crystallographic
    "EDO": ("solvent/additive",  "1,2-Ethanediol (ethylene glycol)"),
    "FMT": ("solvent/additive",  "Formate ion"),
    "ACT": ("solvent/additive",  "Acetate ion"),
    "GOL": ("solvent/additive",  "Glycerol"),
    "SO4": ("solvent/additive",  "Sulfate ion"),
    "PO4": ("solvent/additive",  "Phosphate ion"),
    "PEG": ("solvent/additive",  "Polyethylene glycol"),
    "MPD": ("solvent/additive",  "2-Methyl-2,4-pentanediol"),
    "DMS": ("solvent/additive",  "Dimethyl sulfoxide"),
    "BME": ("solvent/additive",  "Beta-mercaptoethanol"),
    "EPE": ("solvent/additive",  "HEPES"),
    "MES": ("solvent/additive",  "MES buffer"),
    "HED": ("solvent/additive",  "Hydroxyethyl disulfide"),
    "DTT": ("solvent/additive",  "Dithiothreitol"),
    "IMD": ("solvent/additive",  "Imidazole"),
    "ACE": ("solvent/additive",  "Acetyl group"),
    # Ions
    "MG":  ("ion",               "Magnesium(II) ion"),
    "ZN":  ("ion",               "Zinc(II) ion"),
    "CA":  ("ion",               "Calcium(II) ion"),
    "NA":  ("ion",               "Sodium ion"),
    "CL":  ("ion",               "Chloride ion"),
    "K":   ("ion",               "Potassium ion"),
    # VDR primary ligands — secosteroids
    "VDX": ("small-molecule ligand", "1,25-Dihydroxyvitamin D3 (calcitriol)"),
    "VDA": ("small-molecule ligand", "Vitamin D analog"),
    "VDB": ("small-molecule ligand", "Vitamin D analog"),
    "VDZ": ("small-molecule ligand", "Vitamin D analog"),
    "VD1": ("small-molecule ligand", "Vitamin D analog"),
    "VD2": ("small-molecule ligand", "Vitamin D analog"),
    "VD4": ("small-molecule ligand", "Vitamin D analog"),
    "VD5": ("small-molecule ligand", "Vitamin D analog"),
    "VDP": ("small-molecule ligand", "Vitamin D analog"),
    "VHW": ("small-molecule ligand", "Vitamin D analog"),
}

# ── Known solvent/additive codes for auto-classification ─────────────────────
SOLVENT_CODES = {
    "EDO","FMT","ACT","GOL","SO4","PO4","PEG","MPD","DMS","BME",
    "EPE","MES","HED","DTT","IMD","ACE","HOH","WAT","DOD",
}
ION_CODES = {"MG","ZN","CA","NA","CL","K","FE","CU","MN","NI","CO"}


def classify_ligand(code: str) -> tuple[str, str]:
    """Return (ligand_type, proposed_identity) for a given code."""
    if code in LIGAND_TYPE_MAP:
        return LIGAND_TYPE_MAP[code]
    if code in SOLVENT_CODES:
        return ("solvent/additive", code)
    if code in ION_CODES:
        return ("ion", code)
    return ("small-molecule ligand", "VDR ligand — see SMILES")


# ── HTTP session ──────────────────────────────────────────────────────────────
def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "VDR-thesis/2.0 (research)"})
    return session

SESSION = build_session()


# ── RCSB CCD SMILES ───────────────────────────────────────────────────────────
_smiles_cache: dict[str, tuple[str, str, str]] = {}   # code → (smiles, conf, note)

def smiles_from_rcsb(code: str) -> tuple[str, str, str]:
    """
    Returns (smiles, confidence, note).
    Confidence: 'high' if found, 'low' if not.
    """
    code = code.upper()
    if code in _smiles_cache:
        return _smiles_cache[code]

    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
    time.sleep(RCSB_DELAY)
    try:
        resp = SESSION.get(url, timeout=15)
        if resp.status_code == 404:
            result = ("", "low", "not found in RCSB CCD")
            _smiles_cache[code] = result
            return result
        resp.raise_for_status()
        data = resp.json()

        # Try descriptor block first
        desc_block = data.get("rcsb_chem_comp_descriptor") or {}
        smiles = (
            desc_block.get("smiles_stereo") or
            desc_block.get("smiles") or ""
        )
        # Fall back to pdbx_chem_comp_descriptor list
        if not smiles:
            for d in data.get("pdbx_chem_comp_descriptor", []):
                if d.get("type") in ("SMILES_CANONICAL", "SMILES"):
                    smiles = d.get("descriptor", "")
                    break

        if smiles:
            result = (smiles, "high", "RCSB CCD")
        else:
            result = ("", "low", "entry exists in CCD but no SMILES field")

        _smiles_cache[code] = result
        return result

    except Exception as exc:
        result = ("", "low", f"RCSB query error: {exc}")
        _smiles_cache[code] = result
        return result


# ── File scanning ─────────────────────────────────────────────────────────────
def extract_pdb_id(filename: str) -> Optional[str]:
    """
    Extract 4-character PDB ID from filename.
    '3W5P (1).pdb' → '3W5P'
    '3W5P.pdb'     → '3W5P'
    """
    stem = Path(filename).stem          # e.g. '3W5P (1)' or '3W5P'
    m = re.match(r"([A-Za-z0-9]{4})", stem.strip())
    return m.group(1).upper() if m else None


def get_species(pdb_id: str) -> str:
    for sp, ids in SPECIES.items():
        if pdb_id in ids:
            return sp
    return "Unknown"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 68)
    log.info("VDR Ligand Full Pipeline — Rebuild")
    log.info("=" * 68)

    if not DATASET_DIR.exists():
        log.error("Dataset directory not found: %s", DATASET_DIR)
        sys.exit(1)

    # ── Step 1: Scan files ────────────────────────────────────────────────────
    all_files = sorted(
        p for p in DATASET_DIR.iterdir()
        if p.suffix.lower() == ".pdb" and p.is_file()
    )
    log.info("PDB files found: %d", len(all_files))

    unmatched_files = []
    # protein_id → list of Path objects
    protein_files: dict[str, list[Path]] = {}

    for f in all_files:
        pdb_id = extract_pdb_id(f.name)
        if pdb_id and pdb_id in PYMOL_MAP:
            protein_files.setdefault(pdb_id, []).append(f)
        else:
            unmatched_files.append(f.name)

    matched_proteins = len(protein_files)
    log.info("Proteins matched to PyMOL map: %d", matched_proteins)

    if unmatched_files:
        log.warning("Files not matched to any protein ID (%d):", len(unmatched_files))
        for fn in unmatched_files:
            log.warning("  %s", fn)

    # Proteins in PYMOL_MAP but with no file on disk
    all_known = set(PYMOL_MAP.keys())
    proteins_on_disk = set(protein_files.keys())
    missing_from_disk = sorted(all_known - proteins_on_disk)
    if missing_from_disk:
        log.warning(
            "Proteins in PyMOL map but NO file in directory (%d): %s",
            len(missing_from_disk), missing_from_disk,
        )

    # ── Step 2–4: Build rows ──────────────────────────────────────────────────
    rows = []

    # Collect all unique ligand codes to resolve up front (one API call each)
    all_codes = set()
    for codes in PYMOL_MAP.values():
        all_codes.update(codes)

    log.info("\nResolving %d unique ligand codes via RCSB CCD …", len(all_codes))
    for code in sorted(all_codes):
        smiles, conf, note = smiles_from_rcsb(code)
        status = "✓" if smiles else "✗"
        log.info("  %s  %-6s  %s  %s", status, code, conf, note)

    log.info("\nBuilding output table …")

    for pdb_id in sorted(protein_files.keys()):
        files      = protein_files[pdb_id]
        species    = get_species(pdb_id)
        lig_codes  = PYMOL_MAP[pdb_id]          # may have duplicates
        multi_file = len(files) > 1
        multi_lig  = len(set(lig_codes)) > 1

        for file_path in files:
            file_notes = []
            if multi_file:
                file_notes.append(f"multiple ligand files for {pdb_id}")

            # Pair each file with each ligand code
            # Convention: files are indexed same as lig_codes when counts match
            for idx, code in enumerate(lig_codes):
                lig_type, identity = classify_ligand(code)
                smiles, conf, api_note = smiles_from_rcsb(code)

                notes = list(file_notes)
                if multi_lig:
                    notes.append(f"structure contains multiple ligand codes: {lig_codes}")
                if api_note and api_note != "RCSB CCD":
                    notes.append(api_note)
                if len(lig_codes) > 1 and len(files) > 1:
                    notes.append(f"ligand index {idx+1} of {len(lig_codes)}")
                if not smiles:
                    notes.append("SMILES unresolved — manual lookup required")

                rows.append({
                    "Species":           species,
                    "Protein_PDB":       pdb_id,
                    "Ligand_File":       file_path.name,
                    "PyMOL_Ligand_Name": code,
                    "Ligand_Type":       lig_type,
                    "Proposed_Identity": identity,
                    "SMILES":            smiles,
                    "Confidence":        conf if smiles else "low",
                    "Notes":             "; ".join(notes) if notes else "",
                })

    # ── Step 5: Write CSV ─────────────────────────────────────────────────────
    fieldnames = [
        "Species","Protein_PDB","Ligand_File","PyMOL_Ligand_Name",
        "Ligand_Type","Proposed_Identity","SMILES","Confidence","Notes",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    resolved   = sum(1 for r in rows if r["SMILES"])
    unresolved = sum(1 for r in rows if not r["SMILES"])
    high_conf  = sum(1 for r in rows if r["Confidence"] == "high")

    log.info("\n" + "=" * 68)
    log.info("COMPLETE")
    log.info("=" * 68)
    log.info("Output CSV:          %s", OUTPUT_CSV)
    log.info("Total rows:          %d", len(rows))
    log.info("SMILES resolved:     %d  (%.1f%%)", resolved, 100*resolved/max(len(rows),1))
    log.info("SMILES unresolved:   %d", unresolved)
    log.info("High confidence:     %d", high_conf)

    log.info("\nBy species:")
    for sp in ["Human","Rat","Zebrafish","Sea_Lamprey","Unknown"]:
        n = sum(1 for r in rows if r["Species"] == sp)
        if n:
            log.info("  %-14s %d rows", sp, n)

    multi_file_proteins = [p for p, fs in protein_files.items() if len(fs) > 1]
    if multi_file_proteins:
        log.info("\nProteins with multiple ligand files: %s", multi_file_proteins)

    multi_lig_proteins = [p for p, cs in PYMOL_MAP.items() if len(set(cs)) > 1 and p in proteins_on_disk]
    if multi_lig_proteins:
        log.info("Proteins with multiple PyMOL ligand codes: %s", multi_lig_proteins)

    unresolved_codes = sorted({r["PyMOL_Ligand_Name"] for r in rows if not r["SMILES"]})
    if unresolved_codes:
        log.warning("\nUnresolved ligand codes: %s", unresolved_codes)

    # Ligand codes appearing in >1 protein
    from collections import Counter
    code_counts = Counter(
        code for codes in PYMOL_MAP.values() for code in set(codes)
    )
    shared = {c: n for c, n in code_counts.items() if n > 1}
    if shared:
        log.info("\nLigand codes shared across multiple proteins (top 20):")
        for code, n in sorted(shared.items(), key=lambda x: -x[1])[:20]:
            log.info("  %-6s  %d proteins", code, n)


if __name__ == "__main__":
    main()
