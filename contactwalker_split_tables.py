"""
contactwalker_split_tables.py
─────────────────────────────
Converts a ContactWalker full-data CSV (with 3 WT replicates and 3 mutant
replicates) into two analysis tables, handling the ILE129/LEU129 split-residue
merging automatically.

Usage
-----
    python contactwalker_split_tables.py <input_csv> [output_dir]

Arguments
---------
    input_csv   Path to the ContactWalker full-data CSV.
    output_dir  (optional) Directory to write output files.
                Defaults to the same directory as the input file.

Output files
------------
    output_table1_<input_stem>.csv   — 14 columns (WT diffs + WT1-based min change)
    output_table2_<input_stem>.csv   — 18 columns (all 9 WT×mutant pairs + min change)

Expected input columns
----------------------
    Residue_1, Residue_2,
    WT1 (%), WT2 (%), WT3 (%),
    I129F_1 (%), I129F_2 (%), I129F_3 (%),
    Diff_W1_M1, Diff_W1_M2, Diff_W1_M3,
    (other Diff_* columns are ignored)

Wildtype residue label  : ILE129
Mutant residue label    : LEU129
"""

import sys
import os
import pandas as pd
import numpy as np


# ── Column name constants ────────────────────────────────────────────────────
WT_COLS  = ['WT1 (%)', 'WT2 (%)', 'WT3 (%)']
MUT_COLS_IN  = ['I129L_1 (%)', 'I129L_2 (%)', 'I129L_3 (%)']
MUT_COLS_OUT = ['M1 (%)',      'M2 (%)',       'M3 (%)']
WT_LABEL  = 'ILE129'
MUT_LABEL = 'LEU129'


# ── ILE129 / LEU129 merge ────────────────────────────────────────────────────
def merge_split_residue(df, wt_label=WT_LABEL, mut_label=MUT_LABEL):
    """
    The ContactWalker output stores the wildtype residue (ILE129) and the
    mutant residue (LEU129) as separate rows for each contact partner:

      ILE129 row  → WT values non-zero,  mutant values = 0
      LEU129 row  → WT values = 0,       mutant values non-zero

    This function merges them into a single row labelled with the WT residue:
      - WT values  : from the ILE129 row
      - M1/M2/M3   : from the matching LEU129 row

    Three cases handled:
      1. Both ILE129 and LEU129 rows exist for the same partner → merge
      2. Only ILE129 row exists (no LEU129 counterpart)         → keep, M values stay 0
      3. Only LEU129 row exists (no ILE129 counterpart)         → rename to ILE129, WT stays 0
    """
    mask_r1 = df['Residue_1'].isin([wt_label, mut_label])
    mask_r2 = df['Residue_2'].isin([wt_label, mut_label])

    non_129 = df[~mask_r1 & ~mask_r2].copy()

    # Build lookup dicts for LEU129 rows keyed by the partner residue
    leu_r1 = df[df['Residue_1'] == mut_label].set_index('Residue_2')  # LEU129-X
    leu_r2 = df[df['Residue_2'] == mut_label].set_index('Residue_1')  # X-LEU129

    result_rows = []

    # ── Residue_1 = ILE129 ──────────────────────────────────────────────────
    ile_r1_partners = set(df[df['Residue_1'] == wt_label]['Residue_2'])
    for _, row in df[df['Residue_1'] == wt_label].iterrows():
        new_row = row.copy()
        partner = row['Residue_2']
        if partner in leu_r1.index:
            leu = leu_r1.loc[partner]
            for mc in MUT_COLS_OUT:
                new_row[mc] = leu[mc]
        result_rows.append(new_row)

    # LEU129-as-R1 rows with no ILE129 counterpart → rename
    for _, row in df[df['Residue_1'] == mut_label].iterrows():
        if row['Residue_2'] not in ile_r1_partners:
            new_row = row.copy()
            new_row['Residue_1'] = wt_label
            result_rows.append(new_row)

    # ── Residue_2 = ILE129 ──────────────────────────────────────────────────
    ile_r2_partners = set(df[df['Residue_2'] == wt_label]['Residue_1'])
    for _, row in df[df['Residue_2'] == wt_label].iterrows():
        new_row = row.copy()
        partner = row['Residue_1']
        if partner in leu_r2.index:
            leu = leu_r2.loc[partner]
            for mc in MUT_COLS_OUT:
                new_row[mc] = leu[mc]
        result_rows.append(new_row)

    # LEU129-as-R2 rows with no ILE129 counterpart → rename
    for _, row in df[df['Residue_2'] == mut_label].iterrows():
        if row['Residue_1'] not in ile_r2_partners:
            new_row = row.copy()
            new_row['Residue_2'] = wt_label
            result_rows.append(new_row)

    merged_129 = pd.DataFrame(result_rows)
    return pd.concat([non_129, merged_129], ignore_index=True)


# ── Table builders ───────────────────────────────────────────────────────────
def build_table1(df):
    """
    CSV 1 — 14 columns
    WT1-WT2, WT1-WT3 : internal WT variability
    WT1-M1/M2/M3     : WT replicate 1 vs each mutant replicate
    Min Change (WT1) : value with smallest absolute change among WT1-M1/M2/M3
    """
    out = df[['Residue_1', 'Residue_2'] + WT_COLS + MUT_COLS_OUT].copy()

    out['WT1-WT2 (%)'] = df['WT1 (%)'] - df['WT2 (%)']
    out['WT1-WT3 (%)'] = df['WT1 (%)'] - df['WT3 (%)']

    out['WT1-M1 (%)'] = df['WT1 (%)'] - df['M1 (%)']
    out['WT1-M2 (%)'] = df['WT1 (%)'] - df['M2 (%)']
    out['WT1-M3 (%)'] = df['WT1 (%)'] - df['M3 (%)']

    wt1_diffs = out[['WT1-M1 (%)', 'WT1-M2 (%)', 'WT1-M3 (%)']].values
    min_idx   = np.argmin(np.abs(wt1_diffs), axis=1)
    out['Min Change (WT1) (%)'] = wt1_diffs[np.arange(len(wt1_diffs)), min_idx]

    num_cols = out.select_dtypes(include='number').columns
    out[num_cols] = out[num_cols].round(2)
    return out


def build_table2(df):
    """
    CSV 2 — 18 columns
    All 9 WT×mutant pairs: WT1-M1 … WT3-M3
    Min Change (all pairs): value with smallest absolute change across all 9
    """
    out = df[['Residue_1', 'Residue_2'] + WT_COLS + MUT_COLS_OUT].copy()

    diff_col_names = []
    for i, wt_col in enumerate(WT_COLS, start=1):
        for j, m_col in enumerate(MUT_COLS_OUT, start=1):
            col = f'WT{i}-M{j} (%)'
            out[col] = df[wt_col] - df[m_col]
            diff_col_names.append(col)

    all_diffs = out[diff_col_names].values
    min_idx   = np.argmin(np.abs(all_diffs), axis=1)
    out['Min Change (all pairs) (%)'] = all_diffs[np.arange(len(all_diffs)), min_idx]

    num_cols = out.select_dtypes(include='number').columns
    out[num_cols] = out[num_cols].round(2)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main(input_csv, output_dir=None):
    # ── Load ────────────────────────────────────────────────────────────────
    print(f"Loading : {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  {len(df)} rows × {len(df.columns)} columns")

    # ── Validate required columns ────────────────────────────────────────────
    required = ['Residue_1', 'Residue_2'] + WT_COLS + MUT_COLS_IN
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # ── Rename mutant columns to M1/M2/M3 ───────────────────────────────────
    df = df.rename(columns=dict(zip(MUT_COLS_IN, MUT_COLS_OUT)))

    # ── Merge ILE129 / LEU129 rows ───────────────────────────────────────────
    print(f"Merging {WT_LABEL}/{MUT_LABEL} rows ...")
    df = merge_split_residue(df)
    print(f"  {len(df)} rows after merge")

    # ── Build tables ─────────────────────────────────────────────────────────
    t1 = build_table1(df)
    t2 = build_table2(df)

    # ── Output paths ─────────────────────────────────────────────────────────
    stem = os.path.splitext(os.path.basename(input_csv))[0]
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_csv))
    os.makedirs(output_dir, exist_ok=True)

    path1 = os.path.join(output_dir, f"output_table1_{stem}.csv")
    path2 = os.path.join(output_dir, f"output_table2_{stem}.csv")

    t1.to_csv(path1, index=False)
    t2.to_csv(path2, index=False)

    print(f"\nSaved Table 1 → {path1}")
    print(f"  {len(t1)} rows × {len(t1.columns)} columns")
    print(f"  Columns: {t1.columns.tolist()}")

    print(f"\nSaved Table 2 → {path2}")
    print(f"  {len(t2)} rows × {len(t2.columns)} columns")
    print(f"  Columns: {t2.columns.tolist()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_csv  = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    main(input_csv, output_dir)
