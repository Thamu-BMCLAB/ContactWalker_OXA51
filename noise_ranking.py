#!/usr/bin/env python3
"""
Noise Percentile Ranking for Contact Disruption Analysis
========================================================

Sets an empirical noise threshold for ContactWalker contact-disruption
networks.  Designed to pair with contactwalker.py, which produces
the input CSV.


Usage:
    python noise_ranking.py --input contactwalker_full_data.csv \\
        [--output noise_ranking.csv]

Input CSV (output of contactwalker_final.py) must contain:
    - Residue_1, Residue_2
    - WT1 (%), WT2 (%), WT3 (%)  ...  (any number of WT replicates)
    - Mutant columns (auto-detected):
        M1 (%), M2 (%), M3 (%)              (short form)
        mut_1 (%), mut_2 (%), mut_3 (%)     (mut prefix)
        I129V_1 (%), I129V_2 (%)            (mutation-specific, e.g. <LABEL>_n)
"""

import argparse
import re
import numpy as np
import pandas as pd
from itertools import combinations


# ──────────────────────────────────────────────────────────────
# 1. COLUMN AUTO-DETECTION
# ──────────────────────────────────────────────────────────────

def detect_columns(df):
    """Auto-detect WT and Mut replicate occupancy columns.

    WT:  columns matching  WT<n> (%)        e.g. WT1 (%), WT2 (%), WT3 (%)

    Mut (tried in priority order, first match wins):
      1.  M<n> (%)                          e.g. M1 (%), M2 (%)
      2.  mut[_-]?<n> (%)                    e.g. mut_1 (%), mut-2 (%)
      3.  <MUTATION>[_RotX]?[_-]<n> (%)      e.g. I129V_1 (%), I129V_RotA_1 (%)

    Returns (wt_cols, mut_cols), each sorted by replicate number.
    """
    # ── WT ──
    wt_cols = []
    for col in df.columns:
        m = re.match(r'WT(\d+)\s*\(\%\)', col.strip())
        if m:
            wt_cols.append((int(m.group(1)), col))
    wt_cols = [c for _, c in sorted(wt_cols)]

    # ── Mut (priority-ordered patterns) ──
    mut_cols = []

    patterns = [
        r'M(\d+)\s*\(\%\)',                                     # M1 (%)
        r'mut[_\-]?(\d+)\s*\(\%\)',                             # mut_1 (%)
        r'[A-Z]\d{2,3}[A-Z](?:_[A-Za-z]+)?[_\-](\d+)\s*\(\%\)', # I129V_1 (%)
    ]

    for pat in patterns:
        found = []
        for col in df.columns:
            m = re.match(pat, col.strip())
            if m:
                found.append((int(m.group(1)), col))
        if found:
            mut_cols = [c for _, c in sorted(found)]
            break

    return wt_cols, mut_cols


# ──────────────────────────────────────────────────────────────
# 2. MIN-OF-PAIRS NOISE COMPUTATION
# ──────────────────────────────────────────────────────────────

def compute_min_of_pairs_noise(df, rep_cols):
    """Min-of-pairs within-replicate noise for each contact.

    For each contact (row), computes all C(n,2) pairwise absolute
    differences among the n replicates, then takes the MINIMUM.

    This mirrors the Minimal_Change signal (min of 9 WT-MUT pairs),
    so signal and noise are on the same scale.

        3 replicates → 3 pairs → min-of-3
        2 replicates → 1 pair  → that single difference
        1 replicate  → 0 pairs → NaN (noise not computable)

    Returns: numpy array of noise values (NaN where < 2 replicates).
    """
    n_reps = len(rep_cols)
    if n_reps < 2:
        return np.full(len(df), np.nan)

    values = df[rep_cols].values          # (n_contacts, n_reps)
    pair_indices = list(combinations(range(n_reps), 2))

    pair_diffs = np.empty((len(df), len(pair_indices)))
    for k, (i, j) in enumerate(pair_indices):
        pair_diffs[:, k] = np.abs(values[:, i] - values[:, j])

    return pair_diffs.min(axis=1)


# ──────────────────────────────────────────────────────────────
# 3. PERCENTILE COMPUTATION (strictly-below / CDF method)
# ──────────────────────────────────────────────────────────────

def compute_percentile(values):
    """Percentile of each value within the distribution.

    Percentile(v) = (count of values strictly < v) / n × 100

    Properties:
      - Minimum value → percentile 0
      - Tied values   → all share the same percentile
      - Maximum value → percentile < 100 (unless unique)
      - Interpretation: "fraction of the distribution that is below v"

    NaN values are excluded from the distribution and get NaN percentile.
    """
    valid_mask = ~np.isnan(values)
    valid_vals = values[valid_mask]

    if len(valid_vals) == 0:
        return np.full(len(values), np.nan)

    n = len(valid_vals)
    sorted_vals = np.sort(valid_vals)

    # searchsorted(side='left') = count of elements strictly < v
    indices = np.searchsorted(sorted_vals, valid_vals, side='left')
    pct_valid = indices / n * 100

    pct = np.full(len(values), np.nan)
    pct[valid_mask] = pct_valid
    return pct


# ──────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def run_noise_ranking(input_csv, output_csv='noise_ranking.csv'):
    """Compute per-contact noise, percentile-rank, and save CSV.

    Output columns:
        Rank, Residue_1, Residue_2,
        WT_Min_Noise_%, Mut_Min_Noise_%, Overall_Min_Noise_%,
        WT_Percentile, Mut_Percentile, Pooled_Percentile

    Sorted ascending by Pooled_Percentile (Rank 1 = quietest contact).
    """
    # ── Load ──
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} contacts from {input_csv}")

    # ── Detect columns ──
    wt_cols, mut_cols = detect_columns(df)
    print(f"WT replicates  ({len(wt_cols)}): {wt_cols}")
    print(f"Mut replicates ({len(mut_cols)}): {mut_cols}")

    if len(wt_cols) < 2 and len(mut_cols) < 2:
        raise ValueError(
            "Need >= 2 replicates in WT or Mut to compute noise."
        )

    # ── Noise (min-of-pairs, same aggregation as Minimal_Change) ──
    wt_noise = compute_min_of_pairs_noise(df, wt_cols)
    mut_noise = compute_min_of_pairs_noise(df, mut_cols)

    print(f"WT valid noise : {(~np.isnan(wt_noise)).sum()}")
    print(f"Mut valid noise: {(~np.isnan(mut_noise)).sum()}")

    # Overall = min(WT, Mut) — best-case replicate agreement across
    # either condition.  NaN only if BOTH are NaN.
    overall_noise = np.where(
        np.isnan(wt_noise) & np.isnan(mut_noise),
        np.nan,
        np.fmin(wt_noise, mut_noise)
    )

    # ── Percentiles (strictly-below method) ──
    wt_pct = compute_percentile(wt_noise)
    mut_pct = compute_percentile(mut_noise)
    pooled_pct = compute_percentile(overall_noise)

    # ── Build output ──
    result = pd.DataFrame({
        'Residue_1': df['Residue_1'],
        'Residue_2': df['Residue_2'],
        'WT_Min_Noise_%': wt_noise,
        'Mut_Min_Noise_%': mut_noise,
        'Overall_Min_Noise_%': overall_noise,
        'WT_Percentile': wt_pct,
        'Mut_Percentile': mut_pct,
        'Pooled_Percentile': pooled_pct,
    })

    result = result.sort_values(
        'Pooled_Percentile', ascending=True, na_position='last'
    ).reset_index(drop=True)
    result.insert(0, 'Rank', range(1, len(result) + 1))

    result.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}  ({len(result)} contacts)")

    # ── Summary ──
    print(f"\nNoise summary (%):")
    for col in ['WT_Min_Noise_%', 'Mut_Min_Noise_%', 'Overall_Min_Noise_%']:
        v = result[col].dropna()
        if len(v):
            print(f"  {col:22s}  min={v.min():.2f}  median={v.median():.2f}  "
                  f"max={v.max():.2f}  mean={v.mean():.2f}")

    print(f"\nPercentile summary:")
    for col in ['WT_Percentile', 'Mut_Percentile', 'Pooled_Percentile']:
        v = result[col].dropna()
        if len(v):
            print(f"  {col:22s}  min={v.min():.2f}  median={v.median():.2f}  "
                  f"max={v.max():.2f}")

    # ── Threshold reference table ──
    # Pick the noise value at each percentile cutoff to use as --sig
    print(f"\nThreshold reference (use as --sig in network diagram):")
    print(f"  {'Percentile':>12s}  {'WT_noise':>10s}  {'Mut_noise':>10s}  {'Overall':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    wt_v  = wt_noise[~np.isnan(wt_noise)]
    mut_v = mut_noise[~np.isnan(mut_noise)]
    omn_v = overall_noise[~np.isnan(overall_noise)]
    for p in [50, 75, 85, 90, 95, 99]:
        wt_p  = np.percentile(wt_v, p)  if len(wt_v)  else float('nan')
        mut_p = np.percentile(mut_v, p) if len(mut_v) else float('nan')
        omn_p = np.percentile(omn_v, p) if len(omn_v) else float('nan')
        print(f"  P{p:<11d}  {wt_p:>10.2f}  {mut_p:>10.2f}  {omn_p:>10.2f}")

    print(f"\n  Recommended: set --sig to the Overall noise value at your")
    print(f"  chosen percentile (e.g. P95).  Contacts with |Minimal_Change|")
    print(f"  below this value fall within the noise floor and should be")
    print(f"  excluded from the disruption network.")

    return result


# ──────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Noise Percentile Ranking for Contact Disruption Analysis')
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV (contactwalker_full_data.csv)')
    parser.add_argument('--output', '-o', default='noise_ranking.csv',
                        help='Output CSV (default: noise_ranking.csv)')
    args = parser.parse_args()

    run_noise_ranking(args.input, args.output)
