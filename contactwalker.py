

import MDAnalysis as mda
import numpy as np
import pandas as pd
import glob
import multiprocessing as mp
import re

from collections import defaultdict
from tqdm import tqdm
from MDAnalysis.lib.distances import capped_distance


# ============================================================
# USER INPUT — update paths to your PDB snapshot directories
# ============================================================

WT_REPLICAS = {
    "WT1": sorted(glob.glob(
        "/wt1/*.pdb"
    )),
    "WT2": sorted(glob.glob(
        "/wt2/*.pdb"
    )),
    "WT3": sorted(glob.glob(
        "/wt3/*.pdb"
    )),
}

# All three mutant simulations are treated as equivalent I129V replicas.
MUT_REPLICAS = {
    "MUT1": sorted(glob.glob(
        "/mut1/*.pdb"
    )),
    "MUT2": sorted(glob.glob(
        "/mut2/*.pdb"
    )),
    "MUT3": sorted(glob.glob(
        "/mut3/*.pdb"
    )),
}

# ------------------------------------------------------------
# MUTATION SITE NORMALIZATION
# Maps (resid, mutant_resname) → wildtype_resname for pair keys
# so WT and MUT frames share the same dictionary key.
# ------------------------------------------------------------
RESNAME_ALIASES = {
    (129, "VAL"): "ILE",   # I129V: VAL129 → canonical ILE129
}

# ------------------------------------------------------------
# NON-STANDARD RESIDUES INCLUDED IN CONTACT DETECTION
# MDAnalysis's 'protein' selection excludes these modified residues,
# so they are added explicitly via `resname <NAME>` in the selection
# string. KCX = carbamylated lysine (catalytic K70 in OXA-51).
# Add more entries here if your system has other modified residues.
# ------------------------------------------------------------
EXTRA_RESNAMES = ["KCX"]

# ------------------------------------------------------------
# THRESHOLDS
# ------------------------------------------------------------
CONTACT_EXIST_THRESH   = 5.0    # % — contact must exceed this in WT1 or mutant

# ------------------------------------------------------------
# CONTACT CUTOFFS
# ------------------------------------------------------------
CC_CUTOFF    = 5.4   # Carbon-Carbon
OTHER_CUTOFF = 4.6   # All other heavy atoms
MAX_CUTOFF   = max(CC_CUTOFF, OTHER_CUTOFF)

# ------------------------------------------------------------
# CPU SETTINGS
# ------------------------------------------------------------
NCPUS     = 6
CHUNKSIZE = 100

# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------
OUTPUT_FULL    = "contactwalker_full_data.csv"


# ============================================================
# MUTANT LABEL DERIVATION
# ============================================================

# 3-letter → 1-letter amino acid code (for building mutant labels)
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def derive_mutant_label():
    """
    Derive a mutant label like 'I129V' from RESNAME_ALIASES.
    RESNAME_ALIASES maps (resid, mutant_resname) → wildtype_resname,
    so the label is WT_1letter + resid + MUT_1letter.
    e.g. {(129, 'VAL'): 'ILE'} → 'I129V'
    """
    (resid, mut_resname), wt_resname = next(iter(RESNAME_ALIASES.items()))
    return f"{AA3_TO_1[wt_resname]}{resid}{AA3_TO_1[mut_resname]}"


MUTANT_LABEL = derive_mutant_label()   # → "I129V"


# ============================================================
# ATOM SELECTION STRING (includes KCX and other EXTRA_RESNAMES)
# ============================================================

def build_selection():
    """
    Build the atom selection string. MDAnalysis's 'protein' keyword
    excludes non-standard residues like KCX (carbamylated lysine), so
    they are added explicitly with `or resname <NAME>`.
    """
    base = "protein"
    if EXTRA_RESNAMES:
        extra = " or ".join(f"resname {r}" for r in EXTRA_RESNAMES)
        base = f"({base} or {extra})"
    return f"{base} and not name H*"


SELECTION = build_selection()   # e.g. "(protein or resname KCX) and not name H*"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_resid(label):
    """Extract integer residue number from label like 'ARG145'."""
    m = re.search(r"(\d+)", str(label))
    return int(m.group(1)) if m else 0


def normalize_resname(resname, resid):
    """Map mutant residue names at mutation sites to canonical WT name."""
    return RESNAME_ALIASES.get((resid, resname), resname)


def pair_key(res1_name, res1_id, res2_name, res2_id):
    """
    Build canonical, order-independent pair key.
    Normalizes mutation site residue names so WT and MUT frames
    share the same key for the same physical contact.
    """
    res1_name = normalize_resname(res1_name, res1_id)
    res2_name = normalize_resname(res2_name, res2_id)
    if res1_id < res2_id:
        return (f"{res1_name}{res1_id}", f"{res2_name}{res2_id}")
    return (f"{res2_name}{res2_id}", f"{res1_name}{res1_id}")


def min_by_abs(values):
    """
    Return the value with the smallest absolute value from a list,
    retaining its sign.
    e.g. [-30, -10] → -10   [30, 10] → 10   [-30, 10] → 10
    """
    return min(values, key=abs)


# ============================================================
# SINGLE FRAME ANALYSIS
# ============================================================

def analyze_single_pdb(pdb_file):
    contacts = set()
    try:
        u = mda.Universe(pdb_file)
    except Exception as e:
        print(f"Skipping {pdb_file}: {e}")
        return contacts

    protein = u.select_atoms(SELECTION)
    coords        = protein.positions
    atom_resids   = protein.resids
    atom_resnames = protein.resnames
    atom_names    = protein.names
    carbon_mask   = np.char.startswith(atom_names.astype(str), "C")

    try:
        atom_pairs, distances = capped_distance(
            coords, coords, max_cutoff=MAX_CUTOFF,
            return_distances=True, backend="OpenMP")
    except Exception:
        atom_pairs, distances = capped_distance(
            coords, coords, max_cutoff=MAX_CUTOFF,
            return_distances=True)

    for (i, j), dist in zip(atom_pairs, distances):
        if i >= j:
            continue
        resid1 = atom_resids[i]
        resid2 = atom_resids[j]
        if abs(resid1 - resid2) <= 1:
            continue
        if carbon_mask[i] and carbon_mask[j]:
            if dist > CC_CUTOFF:
                continue
        else:
            if dist > OTHER_CUTOFF:
                continue
        contacts.add(pair_key(
            atom_resnames[i], resid1,
            atom_resnames[j], resid2
        ))
    return contacts


# ============================================================
# REPLICA ANALYSIS
# ============================================================

def analyze_replica_parallel(pdb_files, replica_name):
    print(f"\n  Analyzing {replica_name} ({len(pdb_files)} snapshots)...")
    contact_counts = defaultdict(int)
    with mp.Pool(processes=NCPUS) as pool:
        results = list(tqdm(
            pool.imap_unordered(analyze_single_pdb, pdb_files, chunksize=CHUNKSIZE),
            total=len(pdb_files)
        ))
    total_frames = len(results)
    for frame_contacts in results:
        for pair in frame_contacts:
            contact_counts[pair] += 1
    return {pair: 100.0 * count / total_frames
            for pair, count in contact_counts.items()}


def analyze_group(replica_dict):
    return {name: analyze_replica_parallel(files, name)
            for name, files in replica_dict.items()}


# ============================================================
# BUILD FULL DATA TABLE
# ============================================================

def build_full_table(wt_results, mut_results):
    """
    Full data table with all raw occupancies and all pairwise diffs.

    Pairwise diff sign convention: WT - MUT  (matching the ContactWalker paper)
      Negative = mutant GAINS contact (MUT > WT, stabilized in mutant)
      Positive = mutant LOSES contact (MUT < WT, destabilized in mutant)

    Paper example: WT=61%, MUT=98% → WT-MUT = -37% (negative = mutant gains)

    Minimal_Change = min of all 9 pairwise (W1,W2,W3 - M1,M2,M3) by |abs|
    """
    all_pairs = set()
    for rep in wt_results.values():
        all_pairs.update(rep.keys())
    for rep in mut_results.values():
        all_pairs.update(rep.keys())

    sorted_pairs = sorted(
        all_pairs,
        key=lambda x: (extract_resid(x[0]), extract_resid(x[1]))
    )

    wt1  = wt_results["WT1"]
    wt2  = wt_results["WT2"]
    wt3  = wt_results["WT3"]
    mut1 = mut_results["MUT1"]
    mut2 = mut_results["MUT2"]
    mut3 = mut_results["MUT3"]

    # Mutant column names derived from RESNAME_ALIASES (e.g. "I129V_1 (%)")
    c_m1 = f"{MUTANT_LABEL}_1 (%)"
    c_m2 = f"{MUTANT_LABEL}_2 (%)"
    c_m3 = f"{MUTANT_LABEL}_3 (%)"

    rows = []
    for pair in sorted_pairs:
        w1  = wt1.get(pair,  0.0)
        w2  = wt2.get(pair,  0.0)
        w3  = wt3.get(pair,  0.0)
        m1  = mut1.get(pair, 0.0)
        m2  = mut2.get(pair, 0.0)
        m3  = mut3.get(pair, 0.0)

        # WT internal variability: WT1 - WT2, WT1 - WT3
        # (same convention as paper: reference minus compared)
        diff_wt1_wt2 = w1 - w2
        diff_wt1_wt3 = w1 - w3

        # I129V pairwise diffs: WT - MUT (9 combinations)
        # Negative = mutant gains contact; Positive = mutant loses contact
        mut_diffs = [
            w1 - m1, w2 - m1, w3 - m1,
            w1 - m2, w2 - m2, w3 - m2,
            w1 - m3, w2 - m3, w3 - m3,
        ]
        minimal_change = min_by_abs(mut_diffs)

        rows.append({
            "Residue_1":            pair[0],
            "Residue_2":            pair[1],
            # Raw occupancies
            "WT1 (%)":              round(w1, 2),
            "WT2 (%)":              round(w2, 2),
            "WT3 (%)":              round(w3, 2),
            c_m1:                   round(m1, 2),
            c_m2:                   round(m2, 2),
            c_m3:                   round(m3, 2),
            # WT internal variability
            "Diff_WT1_WT2":         round(diff_wt1_wt2, 2),
            "Diff_WT1_WT3":         round(diff_wt1_wt3, 2),
            # All I129V pairwise diffs (WT - MUT)
            "Diff_W1_M1":           round(w1 - m1, 2),
            "Diff_W2_M1":           round(w2 - m1, 2),
            "Diff_W3_M1":           round(w3 - m1, 2),
            "Diff_W1_M2":           round(w1 - m2, 2),
            "Diff_W2_M2":           round(w2 - m2, 2),
            "Diff_W3_M2":           round(w3 - m2, 2),
            "Diff_W1_M3":           round(w1 - m3, 2),
            "Diff_W2_M3":           round(w2 - m3, 2),
            "Diff_W3_M3":           round(w3 - m3, 2),
            # Minimal demonstrated change (min |abs| over all 9 WT-MUT pairs)
            "Minimal_Change":       round(minimal_change, 2),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print(" CONTACTWALKER — FULL DATA TABLE OUTPUT")
    print(f" {MUTANT_LABEL} Mutant (OXA-51)")
    print("="*60)
    print(f" WT1 = most stable WT simulation (W1)")
    print(f" {MUTANT_LABEL} replicas: MUT1, MUT2, MUT3 (treated as equivalent)")
    print(f" Sign convention: WT - MUT  (matches ContactWalker paper)")
    print(f"   Negative = mutant GAINS contact (MUT > WT, stabilized)")
    print(f"   Positive = mutant LOSES contact (MUT < WT, destabilized)")
    print(f" Contact existence threshold : {CONTACT_EXIST_THRESH}%")
    print(f" Atom selection              : {SELECTION}")
    print(f" Extra residues included     : {EXTRA_RESNAMES}")
    print("="*60)

    print("\nMutation site aliases active:")
    for (resid, mut_name), wt_name in RESNAME_ALIASES.items():
        print(f"  {mut_name}{resid} → {wt_name}{resid} (key normalization only)")
    print(f"Derived mutant label: {MUTANT_LABEL}")

    # ── WT Analysis ──────────────────────────────────────────
    print("\n" + "="*60)
    print(" WT ANALYSIS (WT1, WT2, WT3)")
    print("="*60)
    wt_results = analyze_group(WT_REPLICAS)

    # ── Mutant Analysis ──────────────────────────────────────
    print("\n" + "="*60)
    print(f" MUTANT ANALYSIS (MUT1, MUT2, MUT3) — {MUTANT_LABEL}")
    print("="*60)
    mut_results = analyze_group(MUT_REPLICAS)

    # ── Full Data Table ──────────────────────────────────────
    print("\n" + "="*60)
    print(" BUILDING FULL DATA TABLE")
    print("="*60)
    df_full = build_full_table(wt_results, mut_results)
    df_full.to_csv(OUTPUT_FULL, index=False)
    print(f"Full data table : {OUTPUT_FULL}  ({len(df_full)} contact pairs)")

    # Sanity check: residue 129 contacts
    rows_129 = df_full[
        df_full["Residue_1"].str.contains("129") |
        df_full["Residue_2"].str.contains("129")
    ]
    print(f"Sanity check — rows involving residue 129: {len(rows_129)}")

    # Sanity check: KCX70 contacts
    rows_kcx = df_full[
        df_full["Residue_1"].str.contains("KCX") |
        df_full["Residue_2"].str.contains("KCX")
    ]
    print(f"Sanity check — rows involving KCX: {len(rows_kcx)}")
    if len(rows_kcx):
        print(rows_kcx.to_string(index=False))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"  Full data table : {OUTPUT_FULL}  ({len(df_full)} pairs)")
    print(f"  Mutant label    : {MUTANT_LABEL}")
    print(f"  KCX included    : yes (via {SELECTION})")

    print("\nDone.")

    # ── Footnotes reminder ────────────────────────────────────
    print("\n" + "-"*60)
    print("SIGN CONVENTION (for publication):")
    print("  Minimal_Change = WT(%) - MUT(%)")
    print("  Negative = mutant GAINS contact (MUT > WT, stabilized)")
    print("  Positive = mutant LOSES contact (MUT < WT, destabilized)")
    print("  Underline negative values in the published table.")
    print("-"*60)
