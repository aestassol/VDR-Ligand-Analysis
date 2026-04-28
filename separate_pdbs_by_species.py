#!/usr/bin/env python3
"""
Separate PDB structures into species-specific folders.
Uses the verified species mapping from the VDR thesis project.

Run from terminal:
    python3 separate_pdbs_by_species.py

Source folder: /Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/cleared structures/
Output folders created inside that same directory:
    Human/
    Rat/
    Zebrafish/
    Sea_lamprey/
"""

import os
import shutil
import re
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
SOURCE_DIR = Path("/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis/cleared structures")

# ── SPECIES MAPPING (verified, from thesis pipeline) ───────────────────────
SPECIES_MAP = {
    'Human': [
        "1IE8","2HAM","2HAR","2HAS","2HB7","2HB8",
        "3A2I","3A2J","3A3Z","3A40","3AUQ","3AUR","3AX8","3AZ1","3AZ2","3AZ3","3B0T",
        "3CS4","3CS6","3KPZ","3M7R","3OGT","3TKC",
        "3W0A","3W0C","3W0Y","3WGP","3WWR",
        "3X31","3X36",
        "4G2I","5GT4","5V39",
        "8IQN","8IQT",
        "1DB1","1IE9","1S0Z","1S19","1TXI",
        "3A78","3P8X","3VHW",
        "4ITE","4ITF",
        "5YSY","5YT2",
        "7QPP",
        "1YNW",
    ],
    'Rat': [
        "2ZL9","2ZLA","3VT4","3VT5","3VT6","3VT8","3VT9","3VTB","3VTC","3VTD",
        "2ZLC","3VT3","2ZMH","2ZMJ","2ZXM","2ZXN","3AFR","2ZMI",
        "5H1E","5XPL","3VT7",
        "1RJK","1RK3","1RKG","1RKH",
        "2O4J","2O4R","2ZFX",
        "3A2H","3AUN",
        "3VJS","3VJT","3VRT","3VRU","3VRV","3VRW",
        "3W0G","3W0H","3W0I","3W0J",
        "3W5P","3W5Q","3W5R","3W5T",
        "3WT5","3WT6","3WT7","3WTQ",
        "4YNK",
        "5AWJ","5AWK","5B41","5B5B",
        "5GIC","5GID","5GIE",
        "5XPM","5XPN","5XPO","5XPP","5XUQ","5XZF","5XZH",
        "5ZWE","5ZWF","5ZWH","5ZWI",
        "6JEZ","6K5O",
        "7C7V","7C7W","7VQP",
        "9M10","9M11","9M12","9M13","9M14","9M15","9M16","9M17","9M18","9M19",
        "9M1A","9M1B","9M1C","9M1D",
        "9VOG","9VOH","9VOI","9VOJ","9VOK","9VOL",
    ],
    'Zebrafish': [
        "6XZH","6XZI","6XZV","6XZJ","6XZK",
        "3O1D","3O1E",
        "4FHH","4FHI","4IA1","4IA2","4IA3","4IA7","4Q0A","4RUO",
        "5LGA",
        "8P9X","8PWD","8PWE","8PWF","8PWM",
        "8PZ6","8PZ7","8PZ8","8PZ9","8PZB",
        "9EZ1","9EZ2","9FW8",
        "9GY8","9GYA","9GYC","9GYJ","9GYK",
        "9RCG",
        "2HBH","2HC4","2HCD",
        "3DR1",
        "4G1D","4G1Y","4G1Z","4G20","4G21","4G2H",
        "4RUJ","4RUP",
        "5E7V","5MX7","5NKY","5NMA","5NMB",
        "5OW7","5OW9","5OWD",
        "6FO7","6FO8","6FO9","6FOB","6FOD","6T2M",
        "7BNS","7BNU","7BO6",
        "7OXZ","7OY4",
        "7ZFG","7ZFX",
        "8CK5","8CKC","8P9W",
        "9EYR","9FBF",
        "7B39","7OXU",
    ],
    'Sea_lamprey': [
        "7QPI",
    ],
}

# Build reverse lookup: PDB code (uppercase) -> species
CODE_TO_SPECIES = {}
for species, codes in SPECIES_MAP.items():
    for code in codes:
        CODE_TO_SPECIES[code.upper()] = species


def extract_pdb_code(filename):
    """
    Extract the 4-character PDB code from filenames like:
      5OW7.pdb, 5OW7 (1).pdb, 5ow7.pdb, 5OW7_ligand.pdb, etc.
    """
    stem = Path(filename).stem  # remove .pdb
    # Match the first 4 alphanumeric characters (PDB code)
    match = re.match(r'([A-Za-z0-9]{4})', stem)
    if match:
        return match.group(1).upper()
    return None


def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found:\n  {SOURCE_DIR}")
        print("\nPlease verify the path exists on your machine.")
        return

    # Create species subdirectories
    species_dirs = {}
    for species in SPECIES_MAP:
        sp_dir = SOURCE_DIR / species
        sp_dir.mkdir(exist_ok=True)
        species_dirs[species] = sp_dir
        print(f"Created/verified folder: {sp_dir.name}/")

    # Collect all .pdb files (case-insensitive: .pdb, .PDB, .Pdb, etc.)
    pdb_files = [f for f in SOURCE_DIR.iterdir()
                 if f.is_file() and f.suffix.lower() == '.pdb']

    print(f"\nFound {len(pdb_files)} PDB files in source directory.\n")

    # Counters
    moved = {sp: 0 for sp in SPECIES_MAP}
    unmatched = []

    for pdb_path in sorted(pdb_files):
        code = extract_pdb_code(pdb_path.name)  # already returns UPPER
        if code and code.upper() in CODE_TO_SPECIES:
            species = CODE_TO_SPECIES[code.upper()]
            dest = species_dirs[species] / pdb_path.name
            shutil.copy2(pdb_path, dest)   # copy (not move) for safety
            moved[species] += 1
        else:
            unmatched.append(pdb_path.name)

    # ── REPORT ──────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  SPECIES SEPARATION REPORT")
    print("=" * 55)
    total_moved = sum(moved.values())
    for species, count in moved.items():
        print(f"  {species:<14s} : {count:>4d} files")
    print(f"  {'TOTAL SORTED':<14s} : {total_moved:>4d} files")
    print(f"  {'Unmatched':<14s} : {len(unmatched):>4d} files")
    print("=" * 55)

    if unmatched:
        print("\nUnmatched files (not in species mapping):")
        for name in unmatched:
            print(f"  - {name}")

    # Write log file
    log_path = SOURCE_DIR / "species_separation_log.txt"
    with open(log_path, 'w') as log:
        log.write("PDB Species Separation Log\n")
        log.write("=" * 55 + "\n\n")
        for species, count in moved.items():
            log.write(f"{species}: {count} files\n")
        log.write(f"\nTotal sorted: {total_moved}\n")
        log.write(f"Unmatched: {len(unmatched)}\n\n")
        if unmatched:
            log.write("Unmatched files:\n")
            for name in unmatched:
                log.write(f"  {name}\n")
    print(f"\nLog saved to: {log_path.name}")


if __name__ == "__main__":
    main()
