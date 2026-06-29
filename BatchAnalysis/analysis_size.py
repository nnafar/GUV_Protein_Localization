# -*- coding: utf-8 -*-
"""
SIZE ANALYSIS MODULE
Analyzes the physical dimensions (radius) of vesicles across all 4 conditions.

CHANGES FROM PREVIOUS VERSION:
-------------------------------
1. The histogram has been replaced with a combined violin + scatter + box plot.
   This shows:
     - The full distribution shape     (violin — wide where many vesicles)
     - Individual vesicle data points  (scatter / strip plot)
     - Median and quartile lines       (box plot overlaid on the violin)

2. The violin plot is now the only size comparison plot
   (previously there was a separate violin — that's now integrated).

3. NEW: Size statistics and the size plot are now restricted to the
   ANALYSABLE subset (Shape_Quality_Flag == True). Sub-threshold vesicles
   are kept in the master DataFrame (where they carry a phenotype label of
   EXCLUDED, set in analysis_phenotype.py) but are excluded from this
   module's outputs because:
     a) Refined Radius < threshold means membrane detection was unreliable;
        reporting a "size" for these vesicles is misleading.
     b) The size distributions reported in the chapter need to be conditional
        on the analysis threshold to be comparable across conditions.

4. NEW: The per-condition fraction of sub-threshold vesicles is written to
   Size_Excluded_Fractions.csv. This is the left-truncation fraction that
   the chapter's size-statistics paragraph should disclose to be honest
   about what is being summarised.

5. NEW: Size_Statistics_Summary.csv now also reports Median, IQR_25, IQR_75,
   Skewness, and Skewness_Interpretation -- computed on the SAME analysable
   subset as the mean/std/SEM and the violin/box plot itself.

   WHY THIS MATTERS:
   Distribution_Shape_PerCondition.csv (in analysis_distribution_shape.py)
   also reports a radius skewness, but on the FULL, unfiltered population
   (by design -- see that module's docstring). That number does NOT
   describe the same vesicles as the median/IQR reported here, or as the
   violin plot the reader is looking at. Any Results-section sentence that
   reports skewness ALONGSIDE the median/IQR for this figure should cite
   THIS table, not Distribution_Shape_PerCondition.csv, or the two
   statistics in the same sentence will silently refer to two different
   populations.
"""

import pandas as pd
import os
from scipy.stats import skew
import plotting


def _interpret_skewness(s):
    """
    Returns a short human-readable label for a skewness value.
    Same convention as analysis_distribution_shape.py's _interpret_skewness,
    duplicated here (rather than imported) so this module stays
    self-contained, per the project's "no cross-module dependencies between
    analysis modules" pattern.

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


def save_size_statistics(df_analysable, output_dir):
    """
    Calculates descriptive statistics (mean, std, SEM, median, IQR,
    skewness, count) for vesicle radius grouped by condition (Category),
    and saves them to a CSV file.

    Computed on the ANALYSABLE subset only (Shape_Quality_Flag == True) --
    i.e. exactly the vesicles shown in the violin/scatter/box plot. This is
    what makes every number in this CSV directly traceable to that figure.

    Think of this like calculating the average height of students
    in different classes and writing it down -- but only counting
    students whose height could actually be measured properly.
    """
    print("  -> Calculating Size Statistics (analysable vesicles only)...")

    rows = []
    for category, group in df_analysable.groupby('Category'):
        values = group['Refined Radius (um)'].dropna()
        n = len(values)

        if n == 0:
            continue

        row = {
            'Category':           category,
            'Mean_GUV_radius_um': round(float(values.mean()), 4),
            'Std_Dev_um':         round(float(values.std()),  4),
            'SEM_um':             round(float(values.sem()),  4),
            'Median_GUV_radius_um': round(float(values.median()), 4),
            'IQR_25_um':          round(float(values.quantile(0.25)), 4),
            'IQR_75_um':          round(float(values.quantile(0.75)), 4),
            'N_Vesicles':         int(n),
        }

        # Skewness needs a reasonable sample size to be trustworthy.
        # Every condition here has N in the thousands, so MIN_N is a
        # formality, but the guard keeps this safe if ever re-run on a
        # smaller / pilot dataset.
        if n >= 8:
            s = float(skew(values.values, bias=False))
            row['Skewness'] = round(s, 4)
            row['Skewness_Interpretation'] = _interpret_skewness(s)
        else:
            row['Skewness'] = float('nan')
            row['Skewness_Interpretation'] = "n/a"

        rows.append(row)

    stats = pd.DataFrame(rows)

    path = os.path.join(output_dir, "Size_Statistics_Summary.csv")
    stats.to_csv(path, index=False)
    print(f"  -> Size statistics saved: {path}")

    # Console summary so the population-consistent numbers are visible
    # without opening the CSV.
    print("\n      === Size statistics (analysable subset, matches the plot) ===")
    print(stats.to_string(index=False))
    print()

    return stats


def save_excluded_fractions(df_full, output_dir):
    """
    Reports, per condition, what fraction of vesicles fell below the
    analysability threshold (Shape_Quality_Flag == False).

    This is the LEFT-TRUNCATION FRACTION that the chapter's size paragraph
    needs to acknowledge. Without it, the reader cannot tell whether the
    size distributions in the violin plot are conditional on a 5% or a 50%
    truncation -- which matters when comparing conditions.

    Writes a small CSV: Size_Excluded_Fractions.csv
    """
    print("  -> Reporting per-condition excluded fractions...")

    rows = []
    for cat in plotting.CONDITION_ORDER:
        cond = df_full[df_full['Category'] == cat]
        if cond.empty:
            continue
        n_total       = len(cond)
        n_analysable  = cond['Shape_Quality_Flag'].sum()
        n_excluded    = n_total - n_analysable
        pct_excluded  = 100 * n_excluded / n_total if n_total > 0 else 0

        rows.append({
            'Condition':                cat,
            'N_Total_Detected':         int(n_total),
            'N_Analysable':             int(n_analysable),
            'N_Size_Excluded':          int(n_excluded),
            'Pct_Size_Excluded':        round(float(pct_excluded), 2),
        })

    excluded_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Size_Excluded_Fractions.csv")
    excluded_df.to_csv(path, index=False)
    print(f"  -> Excluded fractions saved: {path}")

    # Console summary so this is visible in the run log.
    print("\n      === Size-based exclusion per condition ===")
    print(excluded_df.to_string(index=False))
    print()


def run_size_analysis(df, output_dir):
    """
    Generates the size distribution plot and saves size statistics.

    Workflow
    --------
    1. Report the per-condition excluded fraction (uses the FULL DataFrame).
    2. Filter to the analysable subset.
    3. Plot and summarise on the analysable subset only.

    Parameters
    ----------
    df         : pandas DataFrame -- must contain 'Refined Radius (um)',
                 'Category', and 'Shape_Quality_Flag'
    output_dir : str              -- folder where output files are saved
    """
    print("\n--- Running Size Distribution Analysis ---")

    # ---- Guard against missing flag (back-compat with older CSVs) ------
    if 'Shape_Quality_Flag' not in df.columns:
        print("  ! 'Shape_Quality_Flag' column not found -- "
              "treating all vesicles as analysable.")
        print("  ! Re-run file_handling on the raw data to add the flag.")
        df_full = df.copy()
        df_full['Shape_Quality_Flag'] = True
    else:
        df_full = df

    # ---- 1. Report excluded fractions ----------------------------------
    save_excluded_fractions(df_full, output_dir)

    # ---- 2. Filter to the analysable subset ----------------------------
    df_analysable = df_full[df_full['Shape_Quality_Flag'] == True].copy()

    # ---- 3. Combined violin + scatter + box plot ----------------------
    plotting.plot_violin_scatter_box(
        data=df_analysable,
        x_col='Category',
        y_col='Refined Radius (um)',
        title='GUV Size Distribution by Condition',
        ylabel='GUV radius (µm)',
        output_dir=output_dir,
        filename='Size_Distribution_Violin_Scatter_Box.png',
    )

    # ---- 4. Save descriptive statistics on the analysable subset ------
    save_size_statistics(df_analysable, output_dir)