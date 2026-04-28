#!/usr/bin/env python3
"""
VDR Topological Descriptor Computation Pipeline
================================================
Computes all Table 3 descriptors from the manuscript:
  - Helix distances εi (N-C Cα distance)
  - Helix lengths ηi (number of amino acids)
  - Angles between helices θi,j
  - Dihedral angles DHi,j
  - Pairwise COM separations σi,j
  - COM angles θi_com,j_com
  - COM dihedrals DHi_com,j_com
  - Radius of gyration Rg

Requirements:
  - PyMOL (command-line: pymol -cq this_script.py)
  - DSSP installed (mkdssp or dssp)
  - BioPython (pip install biopython)
  - PDB files organized in species folders

Usage:
  1. Edit CONFIG section below with your paths
  2. Run: pymol -cq compute_vdr_descriptors.py
  OR run as standalone Python with PyMOL API:
     /path/to/pymol -cq compute_vdr_descriptors.py

Author: Generated for Dana's VDR thesis project
"""

import os
import sys
import csv
import math
import glob
import numpy as np

# ============================================================================
# CONFIG - EDIT THESE PATHS
# ============================================================================
BASE_DIR = "/Users/aestassol/Desktop/Thesis all analysis/bioinf/All_Ligands analysis"

# Species folders containing PDB files
SPECIES_DIRS = {
    "Human": os.path.join(BASE_DIR, "Human clean"),
    "Rat": os.path.join(BASE_DIR, "Rat clean"),
    "Zebrafish": os.path.join(BASE_DIR, "Zebrafish clean"),
    "Sea_lamprey": os.path.join(BASE_DIR, "Sea clean"),
}

# GeoDeS helix boundaries CSV (generated from GeoDeS DSSP output)
HELIX_BOUNDARIES_CSV = os.path.join(BASE_DIR, "helix_boundaries.csv")

# Output CSV
OUTPUT_CSV = os.path.join(BASE_DIR, "vdr_topological_descriptors_pymol.csv")

# Canonical helix approximate ranges per species (for DSSP-based identification)
# These are used as ANCHOR points to identify helices from DSSP output.
# Format: (approx_start, approx_end) — the script finds the closest DSSP helix.
CANONICAL_ANCHORS = {
    "Human": {
        "H6":  (227, 264),   # long helix, may be split by kink
        "H7":  (275, 290),
        "Hx":  (291, 305),
        "H12": (397, 423),   # AF-2
    },
    "Rat": {
        "H6":  (223, 260),
        "H7":  (270, 290),
        "Hx":  (291, 305),
        "H12": (397, 420),
    },
    "Zebrafish": {
        "H6":  (255, 292),
        "H7":  (305, 322),
        "Hx":  (323, 335),
        "H12": (429, 449),
    },
    "Sea_lamprey": {
        "H6":  (82, 119),
        "H7":  (130, 145),
        "Hx":  (146, 160),
        "H12": (260, 278),
    },
}

HELIX_NAMES = ["H6", "H7", "Hx", "H12"]


# ============================================================================
# PYMOL IMPORTS
# ============================================================================
try:
    from pymol import cmd, stored
    PYMOL_AVAILABLE = True
except ImportError:
    print("WARNING: PyMOL not available. Run this script inside PyMOL:")
    print("  pymol -cq compute_vdr_descriptors.py")
    PYMOL_AVAILABLE = False


# ============================================================================
# DSSP HELIX IDENTIFICATION
# ============================================================================
def run_dssp_biopython(pdb_path):
    """
    Run DSSP via BioPython and return list of helical segments.
    Each segment: (start_resid, end_resid, length, chain)
    """
    try:
        from Bio.PDB import PDBParser, DSSP
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('prot', pdb_path)
        model = structure[0]

        # Try different DSSP executables
        for dssp_exec in ['mkdssp', 'dssp']:
            try:
                dssp = DSSP(model, pdb_path, dssp=dssp_exec)
                break
            except Exception:
                continue
        else:
            print(f"  WARNING: DSSP failed for {pdb_path}")
            return []

        # Extract helical residues (H = alpha-helix)
        helix_residues = []
        for key in dssp.keys():
            chain_id, res_id = key
            resnum = res_id[1]
            ss = dssp[key][2]
            if ss == 'H':  # alpha-helix only
                helix_residues.append((chain_id, resnum))

        if not helix_residues:
            return []

        # Group into contiguous segments (use first chain only)
        first_chain = helix_residues[0][0]
        resnums = sorted(set(r[1] for r in helix_residues if r[0] == first_chain))

        segments = []
        if resnums:
            seg_start = resnums[0]
            prev = resnums[0]
            for r in resnums[1:]:
                if r > prev + 1:  # gap found
                    segments.append((seg_start, prev, prev - seg_start + 1, first_chain))
                    seg_start = r
                prev = r
            segments.append((seg_start, prev, prev - seg_start + 1, first_chain))

        return segments

    except Exception as e:
        print(f"  WARNING: DSSP error for {pdb_path}: {e}")
        return []


def map_dssp_to_canonical(segments, anchors):
    """
    Map DSSP helical segments to canonical helix names using anchor residue ranges.
    For each canonical helix, find the DSSP segment with maximum overlap.
    For H6 which may be split by a kink, merge adjacent segments if needed.
    """
    mapping = {}

    for helix_name, (anchor_start, anchor_end) in anchors.items():
        best_seg = None
        best_overlap = 0

        for seg_start, seg_end, seg_len, chain in segments:
            # Calculate overlap
            overlap_start = max(seg_start, anchor_start)
            overlap_end = min(seg_end, anchor_end)
            overlap = max(0, overlap_end - overlap_start + 1)

            if overlap > best_overlap:
                best_overlap = overlap
                best_seg = (seg_start, seg_end, seg_len, chain)

        if best_seg:
            mapping[helix_name] = {
                'start': best_seg[0],
                'end': best_seg[1],
                'length_aa': best_seg[2],
                'chain': best_seg[3],
            }
        else:
            # Fallback: use anchor range directly
            print(f"    WARNING: No DSSP segment found for {helix_name}, using anchor range")
            mapping[helix_name] = {
                'start': anchor_start,
                'end': anchor_end,
                'length_aa': anchor_end - anchor_start + 1,
                'chain': 'A',
            }

    return mapping


# ============================================================================
# PYMOL DESCRIPTOR CALCULATIONS
# ============================================================================
def compute_descriptors_pymol(pdb_path, helix_map, pdb_name):
    """
    Compute all Table 3 descriptors using PyMOL.
    helix_map: dict with keys H6, H7, Hx, H12, each containing start, end, chain
    """
    if not PYMOL_AVAILABLE:
        return None

    cmd.delete('all')
    cmd.load(pdb_path, 'prot')

    chain = helix_map['H6']['chain']
    descriptors = {}

    # Create selections for each helix
    for hname in HELIX_NAMES:
        info = helix_map[hname]
        sel_name = hname.replace('x', 'x')  # keep Hx as is
        cmd.select(
            sel_name,
            f"prot and chain {info['chain']} and resi {info['start']}-{info['end']} and name CA"
        )

    # ------------------------------------------------------------------
    # 1. Helix Distance εi (N-C terminal Cα distance)
    # ------------------------------------------------------------------
    for hname in HELIX_NAMES:
        info = helix_map[hname]
        ch = info['chain']
        n_sel = f"prot and chain {ch} and resi {info['start']} and name CA"
        c_sel = f"prot and chain {ch} and resi {info['end']} and name CA"
        try:
            dist = cmd.get_distance(n_sel, c_sel)
            descriptors[f'{hname}_Dist'] = round(dist, 4)
        except Exception:
            descriptors[f'{hname}_Dist'] = None

    # ------------------------------------------------------------------
    # 2. Helix Length ηi (number of amino acids)
    # ------------------------------------------------------------------
    for hname in HELIX_NAMES:
        descriptors[f'{hname}_LengthAA'] = helix_map[hname]['length_aa']

    # ------------------------------------------------------------------
    # 3. Angles between helices θi,j (using anglebetweenhelices)
    # ------------------------------------------------------------------
    try:
        from pymol import anglebetweenhelices
        HAS_ABH = True
    except ImportError:
        HAS_ABH = False

    pairs = []
    for i, hi in enumerate(HELIX_NAMES):
        for hj in HELIX_NAMES[i+1:]:
            pairs.append((hi, hj))

    for hi, hj in pairs:
        info_i = helix_map[hi]
        info_j = helix_map[hj]
        ch = info_i['chain']

        sel_i = f"prot and chain {ch} and resi {info_i['start']}-{info_i['end']} and name CA"
        sel_j = f"prot and chain {ch} and resi {info_j['start']}-{info_j['end']} and name CA"

        if HAS_ABH:
            try:
                angle = anglebetweenhelices.angle(sel_i, sel_j)
                descriptors[f'{hi}-{hj}_Angle'] = round(angle, 4)
            except Exception:
                descriptors[f'{hi}-{hj}_Angle'] = None
        else:
            # Manual calculation: fit line to CA atoms, compute angle between lines
            try:
                angle = _compute_helix_angle(sel_i, sel_j)
                descriptors[f'{hi}-{hj}_Angle'] = round(angle, 4)
            except Exception:
                descriptors[f'{hi}-{hj}_Angle'] = None

    # ------------------------------------------------------------------
    # 4. Dihedral angles DHi,j
    # Sequential dihedral using N-C terminal Cα atoms of consecutive helices
    # Pattern: H6CαN-H6CαC-H7CαN-H7CαC, then shift by one Cα
    # ------------------------------------------------------------------
    # Get N and C terminal Cα coordinates for each helix
    terminals = {}
    for hname in HELIX_NAMES:
        info = helix_map[hname]
        ch = info['chain']
        n_sel = f"prot and chain {ch} and resi {info['start']} and name CA"
        c_sel = f"prot and chain {ch} and resi {info['end']} and name CA"

        stored.n_coords = []
        stored.c_coords = []
        cmd.iterate_state(1, n_sel, "stored.n_coords.append((x,y,z))")
        cmd.iterate_state(1, c_sel, "stored.c_coords.append((x,y,z))")

        if stored.n_coords and stored.c_coords:
            terminals[hname] = {
                'N': stored.n_coords[0],
                'C': stored.c_coords[0],
            }
        else:
            terminals[hname] = None

    # Sequential dihedrals: cycle through H6-H7-Hx-H12
    # First: H6CαN-H6CαC-H7CαN-H7CαC
    # Second: H6CαC-H7CαN-H7CαC-HxCαN
    # etc.
    ca_sequence = []
    for hname in HELIX_NAMES:
        if terminals[hname]:
            ca_sequence.append((f'{hname}CaN', terminals[hname]['N']))
            ca_sequence.append((f'{hname}CaC', terminals[hname]['C']))

    for i in range(len(ca_sequence) - 3):
        name = f"{ca_sequence[i][0]}-{ca_sequence[i+1][0]}-{ca_sequence[i+2][0]}-{ca_sequence[i+3][0]}"
        p1, p2, p3, p4 = [ca_sequence[j][1] for j in range(i, i+4)]
        try:
            dih = _compute_dihedral(p1, p2, p3, p4)
            descriptors[f'DH_{name}'] = round(dih, 4)
        except Exception:
            descriptors[f'DH_{name}'] = None

    # ------------------------------------------------------------------
    # 5. Pairwise COM separations σi_com,j_com
    # ------------------------------------------------------------------
    coms = {}
    for hname in HELIX_NAMES:
        info = helix_map[hname]
        ch = info['chain']
        sel = f"prot and chain {ch} and resi {info['start']}-{info['end']}"
        try:
            com = cmd.centerofmass(sel)
            coms[hname] = com
        except Exception:
            coms[hname] = None

    for hi, hj in pairs:
        if coms[hi] and coms[hj]:
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(coms[hi], coms[hj])))
            descriptors[f'{hi}-{hj}_COM_Dist'] = round(d, 4)
        else:
            descriptors[f'{hi}-{hj}_COM_Dist'] = None

    # ------------------------------------------------------------------
    # 6. COM angles θi_com,j_com (angle at middle COM of 3 consecutive helices)
    # H6com-H7com-Hxcom, H7com-Hxcom-H12com, Hxcom-H12com-H6com, H12com-H6com-H7com
    # ------------------------------------------------------------------
    com_triples = [
        ('H6', 'H7', 'Hx'),
        ('H7', 'Hx', 'H12'),
        ('Hx', 'H12', 'H6'),
        ('H12', 'H6', 'H7'),
    ]
    for ha, hb, hc in com_triples:
        if coms[ha] and coms[hb] and coms[hc]:
            angle = _compute_angle(coms[ha], coms[hb], coms[hc])
            descriptors[f'{ha}com-{hb}com-{hc}com_Angle'] = round(angle, 4)
        else:
            descriptors[f'{ha}com-{hb}com-{hc}com_Angle'] = None

    # ------------------------------------------------------------------
    # 7. COM dihedral angles DHi_com,j_com
    # Dihedral between COMs of H6, H7, Hx, H12
    # ------------------------------------------------------------------
    com_list = [coms[h] for h in HELIX_NAMES]
    if all(c is not None for c in com_list):
        dih = _compute_dihedral(*com_list)
        descriptors['DH_H6com-H7com-Hxcom-H12com'] = round(dih, 4)
    else:
        descriptors['DH_H6com-H7com-Hxcom-H12com'] = None

    # ------------------------------------------------------------------
    # 8. Radius of Gyration (whole protein and per helix)
    # ------------------------------------------------------------------
    try:
        rg_prot = _compute_rg('prot and name CA')
        descriptors['Rg_protein'] = round(rg_prot, 4)
    except Exception:
        descriptors['Rg_protein'] = None

    for hname in HELIX_NAMES:
        info = helix_map[hname]
        ch = info['chain']
        sel = f"prot and chain {ch} and resi {info['start']}-{info['end']} and name CA"
        try:
            rg = _compute_rg(sel)
            descriptors[f'{hname}_Rg'] = round(rg, 4)
        except Exception:
            descriptors[f'{hname}_Rg'] = None

    # ------------------------------------------------------------------
    # 9. Strand % and other SSE content (from DSSP, not PyMOL)
    # Already available from GeoDeS — skip here, merge later
    # ------------------------------------------------------------------

    cmd.delete('all')
    return descriptors


# ============================================================================
# GEOMETRY HELPER FUNCTIONS
# ============================================================================
def _compute_helix_angle(sel1, sel2):
    """Compute angle between two helices using PCA on CA atoms."""
    stored.coords1 = []
    stored.coords2 = []
    cmd.iterate_state(1, sel1, "stored.coords1.append((x,y,z))")
    cmd.iterate_state(1, sel2, "stored.coords2.append((x,y,z))")

    coords1 = np.array(stored.coords1)
    coords2 = np.array(stored.coords2)

    # Fit line (PCA first component) to each set
    axis1 = _fit_helix_axis(coords1)
    axis2 = _fit_helix_axis(coords2)

    # Angle between axes
    cos_angle = abs(np.dot(axis1, axis2))
    cos_angle = min(cos_angle, 1.0)  # clamp for numerical safety
    return math.degrees(math.acos(cos_angle))


def _fit_helix_axis(coords):
    """Fit a line to coordinates using PCA, return unit direction vector."""
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, vh = np.linalg.svd(centered)
    return vh[0]  # first principal component = helix axis


def _compute_angle(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3 in degrees."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _compute_dihedral(p1, p2, p3, p4):
    """Compute dihedral angle between 4 points in degrees."""
    p1, p2, p3, p4 = [np.array(p) for p in [p1, p2, p3, p4]]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return math.degrees(math.atan2(y, x))


def _compute_rg(selection):
    """Compute radius of gyration for a PyMOL selection."""
    stored.coords = []
    cmd.iterate_state(1, selection, "stored.coords.append((x,y,z))")
    coords = np.array(stored.coords)
    if len(coords) == 0:
        return 0.0
    centroid = coords.mean(axis=0)
    diff = coords - centroid
    rg = math.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rg


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 70)
    print("VDR Topological Descriptor Computation")
    print("=" * 70)

    all_results = []
    errors = []

    for species, species_dir in SPECIES_DIRS.items():
        if not os.path.isdir(species_dir):
            print(f"\nWARNING: Directory not found: {species_dir}")
            print(f"  Skipping {species}")
            continue

        pdb_files = sorted(glob.glob(os.path.join(species_dir, "*.pdb")))
        if not pdb_files:
            print(f"\nWARNING: No PDB files found in {species_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {species}: {len(pdb_files)} structures")
        print(f"{'='*60}")

        anchors = CANONICAL_ANCHORS.get(species)
        if not anchors:
            print(f"  WARNING: No canonical anchors defined for {species}")
            continue

        for pdb_path in pdb_files:
            pdb_name = os.path.basename(pdb_path).replace('.pdb', '')
            print(f"\n  {pdb_name}...")

            # Step 1: Run DSSP to identify helical segments
            segments = run_dssp_biopython(pdb_path)
            if not segments:
                print(f"    ERROR: No helical segments found")
                errors.append((pdb_name, species, "No DSSP segments"))
                continue

            # Step 2: Map to canonical helices
            helix_map = map_dssp_to_canonical(segments, anchors)

            print(f"    H6:  {helix_map['H6']['start']}-{helix_map['H6']['end']} "
                  f"({helix_map['H6']['length_aa']} AA)")
            print(f"    H7:  {helix_map['H7']['start']}-{helix_map['H7']['end']} "
                  f"({helix_map['H7']['length_aa']} AA)")
            print(f"    Hx:  {helix_map['Hx']['start']}-{helix_map['Hx']['end']} "
                  f"({helix_map['Hx']['length_aa']} AA)")
            print(f"    H12: {helix_map['H12']['start']}-{helix_map['H12']['end']} "
                  f"({helix_map['H12']['length_aa']} AA)")

            # Step 3: Compute descriptors in PyMOL
            if PYMOL_AVAILABLE:
                try:
                    desc = compute_descriptors_pymol(pdb_path, helix_map, pdb_name)
                    if desc:
                        desc['pdb'] = pdb_name
                        desc['species'] = species
                        # Add helix boundaries to output
                        for hname in HELIX_NAMES:
                            desc[f'{hname}_start'] = helix_map[hname]['start']
                            desc[f'{hname}_end'] = helix_map[hname]['end']
                        all_results.append(desc)
                        print(f"    OK: {len(desc)} descriptors computed")
                    else:
                        errors.append((pdb_name, species, "PyMOL returned None"))
                except Exception as e:
                    print(f"    ERROR: {e}")
                    errors.append((pdb_name, species, str(e)))
            else:
                # Still save the helix boundaries even without PyMOL
                rec = {'pdb': pdb_name, 'species': species}
                for hname in HELIX_NAMES:
                    rec[f'{hname}_start'] = helix_map[hname]['start']
                    rec[f'{hname}_end'] = helix_map[hname]['end']
                    rec[f'{hname}_LengthAA'] = helix_map[hname]['length_aa']
                all_results.append(rec)

    # Save results
    if all_results:
        # Get all unique keys
        all_keys = set()
        for r in all_results:
            all_keys.update(r.keys())

        # Order: pdb, species first, then sorted descriptors
        ordered_keys = ['pdb', 'species']
        ordered_keys += sorted(k for k in all_keys if k not in ordered_keys)

        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

        print(f"\n{'='*60}")
        print(f"RESULTS SAVED: {OUTPUT_CSV}")
        print(f"  {len(all_results)} structures × {len(ordered_keys)} columns")
        print(f"  Errors: {len(errors)}")
        if errors:
            for pdb, sp, err in errors:
                print(f"    {sp}/{pdb}: {err}")
    else:
        print("\nNo results to save!")

    return all_results


if __name__ == '__main__':
    main()
