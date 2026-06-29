# -*- coding: utf-8 -*-
"""
DISTRIBUTION SHAPE ANALYSIS MODULE
===================================

PURPOSE
-------
Quantifies the *shape* of every metric distribution that the summary
statistics module reports a median and IQR for. Two shape moments are
computed per (condition, metric) and per (condition, phenotype, metric):

    SKEWNESS  -- asymmetry of the distribution around its mean.
                  0       = symmetric (e.g. normal distribution)
                  > 0     = right-skewed (long tail toward high values)
                  < 0     = left-skewed  (long tail toward low values)

    KURTOSIS  -- "tailedness" of the distribution (Fisher's definition,
                  i.e. EXCESS kurtosis; a normal distribution = 0).
                  > 0     = heavier tails than normal (more outliers)
                  < 0     = lighter tails than normal (more uniform)

WHY THIS MATTERS
----------------
The summary CSV reports median, mean and IQR but says nothing about
*shape*. Two distributions with identical medians can have very different
biological interpretations:

  Median = 0.5, symmetric  -> typical vesicle close to the median.
  Median = 0.5, skew = -1.3 -> most vesicles HIGH, with a tail of failures.

For *t_cortex* in cortex-forming subsets, skewness routinely exceeds 2
(strong right tail driven by a handful of "thick" outliers). Reporting
shape moments alongside the median IQR makes this visible and protects
the Results text from over-interpreting means.

PROVENANCE
----------
All values in the output CSVs are computed directly from the per-vesicle
master DataFrame using scipy.stats.skew() and scipy.stats.kurtosis().
Both functions use Fisher's definition by default
(normal distribution -> skew = 0, kurtosis = 0).

HOW TO USE
----------
Option A -- called from master_pipeline.py (recommended):
    Add inside main() in master_pipeline.py:

        import analysis_distribution_shape
        analysis_distribution_shape.run_distribution_shape_analysis(
            df, results_dir
        )

    Place AFTER analysis_phenotype.run_phenotype_analysis() so that the
    phenotype-stratified table can be computed.

Option B -- run standalone:
    1. Set STANDALONE_CSV_PATH below to Master_Dataset_Combined.csv
       (NB: must be the version saved AFTER the phenotype step, so the
       Phenotype_Category column is present; otherwise the second CSV
       will simply be skipped).
    2. Run:  python analysis_distribution_shape.py

OUTPUTS (saved to  <results_dir>/Distribution_Shape/ )
-------
    Distribution_Shape_PerCondition.csv
        One row per (condition, metric). Columns:
        Condition, Metric, N, Skewness, Kurtosis,
        Skewness_Interpretation, Kurtosis_Interpretation.

    Distribution_Shape_PerPhenotype.csv
        One row per (condition, phenotype, metric). Same columns plus
        Phenotype_Category. Only produced if 'Phenotype_Category' exists
        in df.

    Distribution_Shape_CortexFormingOnly.csv
        One row per (condition, metric) restricted to vesicles with a
        detectable membrane peak (t_cortex > 0). This is the table that
        the Results section should cite for skewness/kurtosis claims in
        the cortex-forming subset.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# STANDALONE CONFIGURATION
# Only used when you run: python analysis_distribution_shape.py directly.
# Ignored when imported by master_pipeline.py.
# =============================================================================

STANDALONE_CSV_PATH   = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\Output\Batch_Analysis_Results\Master_Dataset_Combined.csv"
STANDALONE_OUTPUT_DIR = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\Output\Batch_Analysis_Results"


# =============================================================================
# METRIC DEFINITIONS
# Mirrors analysis_comparison.py so the same axis labels appear everywhere.
# =============================================================================

METRIC_LABELS = {
    'Refined Radius (um)': 'GUV radius (um)',
    'A localization':      'Localization Score',
    'Gini_Index':          'Gini Index',
    'ISM':                 'ISM',
    't_cortex':            'Cortex Thickness (um)',
    'A Lumen/Bg':          'Actin Lumen / Background',
    'Deformability_Score': 'Deformability Score',
    'Radial_Bumpiness':    'Radial Bumpiness',
    'Sector_Uniformity':   'Sector Uniformity',
    'Clustering_Risk':     'Clustering Risk',
}

# Which conditions have an actin channel. Actin metrics are computed only for
# these (Empty has no actin signal -> all zeros, meaningless skew/kurt).
CONDITIONS_WITH_ACTIN = {'BranchedCortex', 'LinearCortex', 'Factin'}
ACTIN_METRICS = {'A localization', 'Gini_Index', 'ISM', 't_cortex', 'A Lumen/Bg'}

# Condition order for the output CSV. Matches plotting.CONDITION_ORDER but
# we hardcode it here so this module has no dependency on plotting.py.
CONDITION_ORDER = ['Empty', 'Factin', 'BranchedCortex', 'LinearCortex']

# Phenotype display order
PHENOTYPE_ORDER = [
    'Empty', 'Excluded', 'Lumenal', 'Shell',
    'Sparse', 'Patchy', 'Continuous',
]

# Minimum sample size required to compute shape moments reliably.
# scipy will run on N=2 but the values are meaningless.
MIN_N_FOR_SHAPE = 8


# =============================================================================
# INTERPRETATION HELPERS
# =============================================================================

def _interpret_skewness(s):
    """
    Returns a short human-readable label for a skewness value.

    Cut-points follow the standard convention used in textbook statistics:
        |s| < 0.5         -> approximately symmetric
        0.5 <= |s| < 1.0  -> moderately skewed
        |s| >= 1.0        -> highly skewed
    """
    if pd.isna(s):
        return "n/a"
    a = abs(s)
    direction = "right" if s > 0 else "left"
    if a < 0.5:
        return "approximately symmetric"
    if a < 1.0:
        return f"moderately {direction}-skewed"
    return f"highly {direction}-skewed"


def _interpret_kurtosis(k):
    """
    Returns a short human-readable label for an EXCESS kurtosis value
    (Fisher's definition: normal distribution -> 0).

        |k| < 0.5         -> close to normal
        0.5 <= |k| < 3.0  -> moderately heavy/light tailed
        |k| >= 3.0        -> very heavy/light tailed
    """
    if pd.isna(k):
        return "n/a"
    if k > 3.0:
        return "very heavy-tailed"
    if k > 0.5:
        return "moderately heavy-tailed"
    if k < -3.0:
        return "very light-tailed"
    if k < -0.5:
        return "moderately light-tailed"
    return "near-normal tails"


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def _compute_shape(values):
    """
    Returns (N, skewness, kurtosis) for a 1-D numeric array.
    Drops NaN. Uses Fisher's definition for kurtosis (excess kurtosis).
    Returns NaN for skew/kurt if N < MIN_N_FOR_SHAPE.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < MIN_N_FOR_SHAPE:
        return n, np.nan, np.nan
    # bias=False applies the standard sample-statistic correction.
    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, fisher=True, bias=False))
    return n, skew, kurt


def _is_metric_meaningful(condition, metric):
    """
    Returns False if computing shape moments on this (condition, metric)
    combination would be meaningless (e.g. actin metrics on Empty GUVs,
    which are all zero by construction).
    """
    if metric in ACTIN_METRICS and condition not in CONDITIONS_WITH_ACTIN:
        return False
    return True


# =============================================================================
# STEP 1 -- PER-CONDITION TABLE
# =============================================================================

def compute_shape_per_condition(df, output_dir):
    """
    One row per (Condition, Metric). Reports shape moments computed on
    the FULL population of each condition (including zero-cortex GUVs).

    This table is the right reference for size and shape claims, and for
    full-population three-way actin metric claims (matches the
    Comparison_Summary_Statistics.csv that uses the same population).

    Returns the resulting DataFrame.
    """
    print("  -> Computing shape per (condition, metric)...")

    rows = []

    for condition in CONDITION_ORDER:
        if condition not in df['Category'].unique():
            continue

        cond_df = df[df['Category'] == condition]

        for metric, label in METRIC_LABELS.items():
            if metric not in cond_df.columns:
                continue
            if not _is_metric_meaningful(condition, metric):
                continue

            n, skew, kurt = _compute_shape(cond_df[metric].values)
            if n < MIN_N_FOR_SHAPE:
                continue

            rows.append({
                'Condition':               condition,
                'Metric':                  label,
                'N':                       int(n),
                'Skewness':                round(skew, 3),
                'Kurtosis':                round(kurt, 3),
                'Skewness_Interpretation': _interpret_skewness(skew),
                'Kurtosis_Interpretation': _interpret_kurtosis(kurt),
            })

    out_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Distribution_Shape_PerCondition.csv")
    out_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")
    return out_df


# =============================================================================
# STEP 2 -- CORTEX-FORMING SUBSET
# =============================================================================

def compute_shape_cortex_only(df, output_dir):
    """
    Same as compute_shape_per_condition BUT restricted to vesicles with a
    detectable membrane peak (t_cortex > 0). This mirrors the
    Comparison_Summary_Statistics_CortexOnly.csv subset used by the
    phenotype-stratified comparison in the Results section.

    The Empty condition is dropped (no t_cortex defined).

    Returns the resulting DataFrame.
    """
    print("  -> Computing shape on cortex-forming subset (t_cortex > 0)...")

    if 't_cortex' not in df.columns:
        print("      ! 't_cortex' column not found; skipping cortex-only table.")
        return pd.DataFrame()

    rows = []

    for condition in CONDITION_ORDER:
        if condition == 'Empty':
            continue
        if condition not in df['Category'].unique():
            continue

        sub = df[(df['Category'] == condition) & (df['t_cortex'] > 0)]
        if len(sub) < MIN_N_FOR_SHAPE:
            continue

        for metric, label in METRIC_LABELS.items():
            if metric not in sub.columns:
                continue
            if not _is_metric_meaningful(condition, metric):
                continue

            n, skew, kurt = _compute_shape(sub[metric].values)
            if n < MIN_N_FOR_SHAPE:
                continue

            rows.append({
                'Condition':               condition,
                'Subset':                  'CortexForming (t_cortex>0)',
                'Metric':                  label,
                'N':                       int(n),
                'Skewness':                round(skew, 3),
                'Kurtosis':                round(kurt, 3),
                'Skewness_Interpretation': _interpret_skewness(skew),
                'Kurtosis_Interpretation': _interpret_kurtosis(kurt),
            })

    out_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Distribution_Shape_CortexFormingOnly.csv")
    out_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")
    return out_df


# =============================================================================
# STEP 3 -- PER-PHENOTYPE TABLE
# =============================================================================

def compute_shape_per_phenotype(df, output_dir):
    """
    One row per (Condition, Phenotype, Metric). Only runs if the
    phenotype column exists. Skips phenotype groups with N < MIN_N_FOR_SHAPE
    (so e.g. tiny EXCLUDED groups won't bloat the table).

    Returns the resulting DataFrame (empty if Phenotype_Category is absent).
    """
    if 'Phenotype_Category' not in df.columns:
        print("  ! 'Phenotype_Category' not found; "
              "run analysis_phenotype first. Skipping per-phenotype table.")
        return pd.DataFrame()

    print("  -> Computing shape per (condition, phenotype, metric)...")

    rows = []

    for condition in CONDITION_ORDER:
        if condition not in df['Category'].unique():
            continue

        cond_df = df[df['Category'] == condition]
        phenotypes_here = [
            p for p in PHENOTYPE_ORDER if p in cond_df['Phenotype_Category'].unique()
        ]

        for phenotype in phenotypes_here:
            sub = cond_df[cond_df['Phenotype_Category'] == phenotype]
            if len(sub) < MIN_N_FOR_SHAPE:
                continue

            for metric, label in METRIC_LABELS.items():
                if metric not in sub.columns:
                    continue
                if not _is_metric_meaningful(condition, metric):
                    continue

                n, skew, kurt = _compute_shape(sub[metric].values)
                if n < MIN_N_FOR_SHAPE:
                    continue

                rows.append({
                    'Condition':               condition,
                    'Phenotype_Category':      phenotype,
                    'Metric':                  label,
                    'N':                       int(n),
                    'Skewness':                round(skew, 3),
                    'Kurtosis':                round(kurt, 3),
                    'Skewness_Interpretation': _interpret_skewness(skew),
                    'Kurtosis_Interpretation': _interpret_kurtosis(kurt),
                })

    out_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Distribution_Shape_PerPhenotype.csv")
    out_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")
    return out_df


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def _print_summary(per_cond_df, cortex_df):
    """
    Prints a short readable summary to the terminal so the user can see at
    a glance which distributions are highly skewed without opening the CSV.
    """
    if per_cond_df.empty:
        return

    print("\n      === Distribution shape: per condition (full population) ===")
    print(f"      {'Condition':<16}{'Metric':<28}{'N':>7}{'Skew':>9}{'Kurt':>9}   Note")
    print("      " + "-" * 90)
    for _, row in per_cond_df.iterrows():
        flag = ""
        if abs(row['Skewness']) >= 1.0:
            flag = "  <-- highly skewed"
        elif row['Kurtosis'] >= 3.0:
            flag = "  <-- heavy tails"
        print(
            f"      {row['Condition']:<16}{row['Metric']:<28}"
            f"{int(row['N']):>7}{row['Skewness']:>9.2f}{row['Kurtosis']:>9.2f}"
            f"{flag}"
        )

    if not cortex_df.empty:
        print("\n      === Distribution shape: cortex-forming subset (t_cortex > 0) ===")
        print(f"      {'Condition':<16}{'Metric':<28}{'N':>7}{'Skew':>9}{'Kurt':>9}   Note")
        print("      " + "-" * 90)
        for _, row in cortex_df.iterrows():
            flag = ""
            if abs(row['Skewness']) >= 1.0:
                flag = "  <-- highly skewed"
            elif row['Kurtosis'] >= 3.0:
                flag = "  <-- heavy tails"
            print(
                f"      {row['Condition']:<16}{row['Metric']:<28}"
                f"{int(row['N']):>7}{row['Skewness']:>9.2f}{row['Kurtosis']:>9.2f}"
                f"{flag}"
            )
    print()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_distribution_shape_analysis(df, output_dir):
    """
    Runs all three shape-moment tables in sequence.

    Parameters
    ----------
    df         : master DataFrame from file_handling.load_and_process_data()
                 (ideally AFTER analysis_phenotype has added
                 'Phenotype_Category', so the per-phenotype table is built).
    output_dir : Batch_Analysis_Results folder path.

    Outputs
    -------
    <output_dir>/Distribution_Shape/Distribution_Shape_PerCondition.csv
    <output_dir>/Distribution_Shape/Distribution_Shape_CortexFormingOnly.csv
    <output_dir>/Distribution_Shape/Distribution_Shape_PerPhenotype.csv
        (last one only if 'Phenotype_Category' is present)
    """
    print("\n--- Running Distribution Shape Analysis ---")

    shape_dir = os.path.join(output_dir, "Distribution_Shape")
    os.makedirs(shape_dir, exist_ok=True)

    if 'Category' not in df.columns:
        print("  ! 'Category' column not found. Cannot run shape analysis.")
        return

    per_cond_df  = compute_shape_per_condition(df, shape_dir)
    cortex_df    = compute_shape_cortex_only(df, shape_dir)
    _per_pheno   = compute_shape_per_phenotype(df, shape_dir)   # noqa: F841

    _print_summary(per_cond_df, cortex_df)

    print("  -> Distribution Shape Analysis Complete.\n")


# =============================================================================
# STANDALONE MODE
# =============================================================================

if __name__ == "__main__":
    """
    Runs when you execute:  python analysis_distribution_shape.py

    Does NOT run when master_pipeline.py does:
        import analysis_distribution_shape
    """
    print("Running analysis_distribution_shape.py in standalone mode...")

    if not os.path.exists(STANDALONE_CSV_PATH):
        print(f"\nERROR: Could not find CSV at:\n  {STANDALONE_CSV_PATH}")
        print("Update STANDALONE_CSV_PATH at the top of this file,")
        print("or run master_pipeline.py first to generate the master CSV.\n")
    else:
        df  = pd.read_csv(STANDALONE_CSV_PATH)
        out = STANDALONE_OUTPUT_DIR or os.path.dirname(STANDALONE_CSV_PATH)
        print(f"Loaded {len(df)} vesicles.\n")
        run_distribution_shape_analysis(df, out)
        print("Done.")