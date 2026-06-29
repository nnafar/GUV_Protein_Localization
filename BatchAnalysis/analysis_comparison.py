# -*- coding: utf-8 -*-
"""
CROSS-CONDITION COMPARISON MODULE
===================================

PURPOSE
-------
This module answers biological questions about your GUV data by comparing
conditions systematically.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")       # save to file; don't try to open a screen window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu

import plotting             # your existing plotting module (colours, styles)


# =============================================================================
# STANDALONE CONFIGURATION
# Only used when you run: python analysis_comparison.py directly.
# Ignored when imported by master_pipeline.py.
# =============================================================================

STANDALONE_CSV_PATH   = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\Output\Batch_Analysis_Results\Master_Dataset_Combined.csv"
STANDALONE_OUTPUT_DIR = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\Output\Batch_Analysis_Results"

# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

# Every metric we might compare, with the axis label that will appear on plots.
# Key  = exact column name in the CSV
# Value = human-readable label printed on the plot axis
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

# Metrics that only make sense for conditions WITH an actin channel.
# (Empty has no actin, so actin metrics on Empty are meaningless noise.)
ACTIN_METRICS = ['A localization', 'Gini_Index', 'ISM', 't_cortex']

# Lumen/background ratio -- measures how much actin leaks inside the GUV.
# Only meaningful for actin conditions.
LUMEN_METRICS = ['A Lumen/Bg']

# Shape quality metrics -- meaningful for ALL conditions, because they come
# from the membrane channel, not the actin channel.
SHAPE_METRICS = ['Deformability_Score', 'Radial_Bumpiness', 'Sector_Uniformity']

# Which conditions have an actin channel
CONDITIONS_WITH_ACTIN = {'BranchedCortex', 'LinearCortex', 'Factin'}

# The three actin conditions in order for 3-way comparisons
ACTIN_ORDER = ['Factin', 'BranchedCortex', 'LinearCortex']


# =============================================================================
# SHARED HELPERS  (statistics only — plots live in plotting.py)
# =============================================================================

def _mw_stars_local(vals_a, vals_b):
    """
    Runs a Mann-Whitney U test and returns stars + p-value.
    Used internally by run_comparison_tests() for the CSV output.
    The visual equivalent (_mw_stars) lives in plotting.py.
    """
    if len(vals_a) < 5 or len(vals_b) < 5:
        return "ns", 1.0
    try:
        _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        stars = ("***" if p < 0.001 else ("**" if p < 0.01 else
                 ("*"  if p < 0.05  else "ns")))
        return stars, p
    except Exception:
        return "ns", 1.0


def _cliffs_delta(a, b):
    """
    Calculates Cliff's Delta: how large is the difference between two groups?

    The p-value tells you "is this difference real?"
    Cliff's Delta tells you "how big is it in practice?"
      0.00 – 0.15  → Negligible
      0.15 – 0.33  → Small
      0.33 – 0.47  → Medium
      0.47+        → Large

    Returns a value between 0.0 and 1.0 (we use the absolute value).
    """
    a, b = np.sort(a), np.sort(b)
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0

    j = k = more = less = 0
    for x in a:
        while j < n and b[j] < x:
            j += 1
        while k < n and b[k] <= x:
            k += 1
        more += j
        less += (n - k)

    return abs((more - less) / (m * n))


def _effect_label(d):
    """Converts a Cliff's Delta value to a human-readable size label."""
    if d < 0.147: return "Negligible"
    if d < 0.330: return "Small"
    if d < 0.474: return "Medium"
    return "Large"


# =============================================================================
# STEP 1 -- SUMMARY STATISTICS TABLE
# =============================================================================

def compute_summary_statistics(df, output_dir):
    """
    Computes a summary table: for every condition x metric combination,
    calculates N, median, mean, standard deviation, and IQR.

    WHAT IS IQR?
    ------------
    IQR = Inter-Quartile Range. The distance between the 25th and 75th
    percentile -- the range containing the middle 50% of your data.
    A small IQR means most vesicles are similar.
    A large IQR means high spread.

    ANALOGY: If exam scores range from 20-100 but 50% of students scored
    between 60-75, the IQR is 15. It tells you where most people landed,
    ignoring the extremes at either end.

    Parameters
    ----------
    df         : the master DataFrame (all vesicles, all conditions)
    output_dir : folder where the CSV will be saved

    Returns
    -------
    summary_df : a DataFrame with one row per (condition, metric)
    """
    print("  -> Computing cross-condition summary statistics...")

    rows = []

    for condition in plotting.CONDITION_ORDER:

        if condition not in df['Category'].unique():
            continue

        cond_df = df[df['Category'] == condition].copy()

        for metric, label in METRIC_LABELS.items():

            if metric not in cond_df.columns:
                continue

            values = cond_df[metric].dropna()

            if len(values) < 3:
                continue

            rows.append({
                'Condition': condition,
                'Metric':    label,
                'N':         int(len(values)),
                'Median':    round(float(values.median()),       3),
                'Mean':      round(float(values.mean()),         3),
                'Std_Dev':   round(float(values.std()),          3),
                'IQR_25':    round(float(values.quantile(0.25)), 3),
                'IQR_75':    round(float(values.quantile(0.75)), 3),
            })

    summary_df = pd.DataFrame(rows)

    path = os.path.join(output_dir, "Comparison_Summary_Statistics.csv")
    summary_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    # Print a pivot table (rows = conditions, columns = metrics) to the console
    print("\n      === Summary: Median values per condition ===")
    try:
        pivot = summary_df.pivot_table(
            index='Condition', columns='Metric', values='Median'
        )
        pivot = pivot.reindex([
            c for c in plotting.CONDITION_ORDER if c in pivot.index
        ])
        print(pivot.to_string())
    except Exception:
        print(summary_df.to_string())
    print()

    return summary_df


# =============================================================================
# STEP 2 -- STATISTICAL TESTS
# =============================================================================

def run_comparison_tests(df, output_dir):
    """
    Runs pairwise statistical tests for all comparison groups.

    Covers five groups:
      A. Radius:               each cortex condition vs Empty
      B. Actin metrics:        BranchedCortex vs LinearCortex
                               BranchedCortex vs Factin
                               LinearCortex   vs Factin   (three-way)
      C. Lumen retention:      three-way on A Lumen/Bg
      D. Shape metrics:        all four conditions pairwise
      E. Phenotype-stratified: BranchedCortex vs LinearCortex,
                               CONTINUOUS vesicles only

    Each test produces:
      - p-value (Mann-Whitney U)
      - Cliff's Delta + label (effect size)
      - Median difference (in real units -- useful for biological interpretation)

    Parameters
    ----------
    df         : master DataFrame
    output_dir : save folder

    Returns
    -------
    results_df : DataFrame of all test results
    """
    print("  -> Running pairwise comparison tests...")

    results = []

    # ---- Internal helper ------------------------------------------------
    def _test(label_a, vals_a, label_b, vals_b, metric, group=''):
        """
        Runs one Mann-Whitney test and appends the result to `results`.

        This is a 'closure' -- a function defined inside another function.
        It can see and modify the `results` list from the outer function.
        Think of it as a personal assistant who knows to file results in
        the right drawer without being told each time.
        """
        if len(vals_a) < 5 or len(vals_b) < 5:
            return

        try:
            _, p_val = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            d_val    = _cliffs_delta(vals_a, vals_b)
            stars    = ("***" if p_val < 0.001 else ("**" if p_val < 0.01 else
                        ("*"  if p_val < 0.05  else "ns")))

            results.append({
                'Analysis_Group':    group,
                'Comparison':        f"{label_a}  vs  {label_b}",
                'Metric':            METRIC_LABELS.get(metric, metric),
                'N_A':               len(vals_a),
                'Median_A':          round(float(np.median(vals_a)), 3),
                'N_B':               len(vals_b),
                'Median_B':          round(float(np.median(vals_b)), 3),
                'Median_Difference': round(float(np.median(vals_a) - np.median(vals_b)), 3),
                'p_value':           round(float(p_val), 6),
                'Significance':      stars,
                'Cliffs_Delta':      round(float(d_val), 4),
                'Effect_Size':       _effect_label(d_val),
            })

        except Exception as e:
            print(f"      ! Test failed ({label_a} vs {label_b}, {metric}): {e}")


    # ---- A. Radius: each cortex condition vs Empty ----------------------
    print("      A. Radius: cortex vs empty...")

    empty_radius = df[df['Category'] == 'Empty']['Refined Radius (um)'].dropna().values

    for cond in ['Factin', 'BranchedCortex', 'LinearCortex']:
        cond_radius = df[df['Category'] == cond]['Refined Radius (um)'].dropna().values
        _test(
            label_a = plotting.CONDITION_LABELS.get(cond, cond),
            vals_a  = cond_radius,
            label_b = "Empty",
            vals_b  = empty_radius,
            metric  = 'Refined Radius (um)',
            group   = 'A. Radius vs Empty',
        )


    # ---- B. Actin metrics: three-way (Factin / Branched / Linear) -------
    print("      B. Actin metrics: Factin / Branched / Linear (three-way)...")

    factin_df   = df[df['Category'] == 'Factin']
    branched_df = df[df['Category'] == 'BranchedCortex']
    linear_df   = df[df['Category'] == 'LinearCortex']

    for metric in ACTIN_METRICS:
        if metric not in df.columns:
            continue
        f_vals = factin_df[metric].dropna().values
        b_vals = branched_df[metric].dropna().values
        l_vals = linear_df[metric].dropna().values

        _test("Branched", b_vals, "Linear",  l_vals, metric, 'B. Actin Three-Way')
        _test("Branched", b_vals, "F-actin", f_vals, metric, 'B. Actin Three-Way')
        _test("Linear",   l_vals, "F-actin", f_vals, metric, 'B. Actin Three-Way')


    # ---- C. Lumen retention: A Lumen/Bg three-way ----------------------
    # A Lumen/Bg = how much actin ended up INSIDE the GUV.
    # Lower = better for a clean cortex (actin stays at the membrane).
    print("      C. Lumen retention (A Lumen/Bg)...")

    metric = 'A Lumen/Bg'
    if metric in df.columns:
        f_lum = factin_df[metric].dropna().values
        b_lum = branched_df[metric].dropna().values
        l_lum = linear_df[metric].dropna().values

        _test("Branched", b_lum, "Linear",  l_lum, metric, 'C. Lumen Retention')
        _test("Branched", b_lum, "F-actin", f_lum, metric, 'C. Lumen Retention')
        _test("Linear",   l_lum, "F-actin", f_lum, metric, 'C. Lumen Retention')


    # ---- D. Shape metrics: all 4 conditions pairwise -------------------
    # Shape metrics use the membrane channel, so Empty is a valid group here.
    print("      D. Shape metrics: all 4 conditions...")

    all_conditions = ['Empty', 'Factin', 'BranchedCortex', 'LinearCortex']

    for metric in SHAPE_METRICS:
        if metric not in df.columns:
            continue
        # Compare every unique pair (i < j avoids testing A vs B AND B vs A)
        for i, cond_a in enumerate(all_conditions):
            for cond_b in all_conditions[i + 1:]:
                vals_a = df[df['Category'] == cond_a][metric].dropna().values
                vals_b = df[df['Category'] == cond_b][metric].dropna().values
                _test(
                    plotting.CONDITION_LABELS.get(cond_a, cond_a), vals_a,
                    plotting.CONDITION_LABELS.get(cond_b, cond_b), vals_b,
                    metric, 'D. Shape Metrics',
                )


    # ---- E. Phenotype-stratified: CONTINUOUS only ----------------------
    # WHY STRATIFY BY PHENOTYPE?
    # --------------------------
    # If BranchedCortex has 70% CONTINUOUS vesicles and LinearCortex has 30%,
    # comparing all vesicles of each condition mixes:
    #   Effect 1 -- cortex type  (what we want to measure)
    #   Effect 2 -- phenotype distribution difference  (a confound)
    #
    # Keeping only CONTINUOUS vesicles from both groups controls for Effect 2.
    # Any remaining difference is then more likely due to cortex type alone.
    print("      E. Phenotype-stratified (CONTINUOUS only)...")

    if 'Phenotype_Category' in df.columns:
        b_cont = branched_df[branched_df['Phenotype_Category'] == 'Continuous']
        l_cont = linear_df[linear_df['Phenotype_Category'] == 'Continuous']

        for metric in ACTIN_METRICS + LUMEN_METRICS:
            if metric not in df.columns:
                continue
            b_vals = b_cont[metric].dropna().values
            l_vals = l_cont[metric].dropna().values
            _test(
                "Branched (CONTINUOUS)", b_vals,
                "Linear (CONTINUOUS)",   l_vals,
                metric, 'E. Phenotype-Stratified (CONTINUOUS)',
            )
    else:
        print("      ! 'Phenotype_Category' not found -- "
              "run analysis_phenotype before analysis_comparison.")


    # ---- Save and print ------------------------------------------------
    if not results:
        print("      -> No valid comparisons found.")
        return pd.DataFrame()

    results_df = (pd.DataFrame(results)
                  .sort_values(['Analysis_Group', 'Metric', 'p_value']))

    path = os.path.join(output_dir, "Comparison_Statistical_Tests.csv")
    results_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    # Console summary -- star = significant, tick = not significant
    print("\n      === Statistical Test Results ===")
    for group, grp_df in results_df.groupby('Analysis_Group'):
        print(f"\n      [{group}]")
        for _, row in grp_df.iterrows():
            flag = "  *" if row['p_value'] < 0.05 else "  -"
            print(
                f"{flag}  {row['Comparison']:<45}  "
                f"{row['Metric']:<28}  "
                f"p={row['p_value']:.4f} {row['Significance']:>3}  "
                f"d={row['Cliffs_Delta']:.3f} ({row['Effect_Size']})"
            )
    print()

    return results_df


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_comparison_analysis(df, output_dir):
    """
    Runs ALL comparison analyses in sequence.
    Call this from master_pipeline.py, or use standalone mode below.

    Steps
    -----
    1.  Create Comparison_Analysis sub-folder
    2.  Summary statistics CSV
    3.  Statistical tests CSV  (5 analysis groups)
    4.  Plot A: Radius (all conditions)        → plotting.plot_radius_all_conditions
    5.  Plot B: Radius (cortex vs empty)       → plotting.plot_cortex_vs_empty_radius
    6.  Plot C: Actin metrics three-way        → plotting.plot_actin_three_way
    7.  Plot D: Lumen retention (A Lumen/Bg)   → plotting.plot_lumen_retention
    8.  Plot E: Phenotype-stratified (CONT.)   → plotting.plot_phenotype_stratified

    All plot functions live in plotting.py (Section 12).
    This function handles only orchestration and CSV generation.

    Parameters
    ----------
    df         : master DataFrame from file_handling.load_and_process_data()
    output_dir : Batch_Analysis_Results folder path
    """
    print("\n--- Running Cross-Condition Comparison Analysis ---")

    comp_dir = os.path.join(output_dir, "Comparison_Analysis")
    os.makedirs(comp_dir, exist_ok=True)

    plotting.set_paper_style()

    # Minimum column check
    if 'Category' not in df.columns:
        print("  ! 'Category' column not found. Cannot run comparison analysis.")
        return
    if 'Refined Radius (um)' not in df.columns:
        print("  ! Radius column not found. Check that pipeline ran correctly.")
        return

    # ── CSV outputs (statistics stay in this module) ─────────────────────────
    compute_summary_statistics(df, comp_dir)
    run_comparison_tests(df, comp_dir)

    # ── Plot outputs (all visual functions live in plotting.py) ───────────────
    plotting.plot_radius_all_conditions(df, comp_dir)
    plotting.plot_cortex_vs_empty_radius(df, comp_dir)
    plotting.plot_actin_three_way(df, comp_dir)
    plotting.plot_lumen_retention(df, comp_dir)
    plotting.plot_phenotype_stratified(df, comp_dir)

    print("  -> Comparison Analysis Complete.\n")


# =============================================================================
# STANDALONE MODE
# =============================================================================

if __name__ == "__main__":
    """
    Runs when you execute:  python analysis_comparison.py

    Does NOT run when master_pipeline.py does:  import analysis_comparison
    (because then __name__ is 'analysis_comparison', not '__main__')
    """
    print("Running analysis_comparison.py in standalone mode...")

    if not os.path.exists(STANDALONE_CSV_PATH):
        print(f"\nERROR: Could not find CSV at:\n  {STANDALONE_CSV_PATH}")
        print("Update STANDALONE_CSV_PATH at the top of this file,")
        print("or run master_pipeline.py first to generate the master CSV.\n")
    else:
        df  = pd.read_csv(STANDALONE_CSV_PATH)
        out = STANDALONE_OUTPUT_DIR or os.path.dirname(STANDALONE_CSV_PATH)
        print(f"Loaded {len(df)} vesicles.\n")
        run_comparison_analysis(df, out)
        print("Done.")