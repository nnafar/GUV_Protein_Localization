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
"""

import pandas as pd
import os
import plotting


def save_size_statistics(df_analysable, output_dir):
    """
    Calculates descriptive statistics (mean, std, SEM, count) for vesicle radius
    grouped by condition (Category), and saves them to a CSV file.

    Computed on the ANALYSABLE subset only (Shape_Quality_Flag == True).

    Think of this like calculating the average height of students
    in different classes and writing it down -- but only counting
    students whose height could actually be measured properly.
    """
    print("  -> Calculating Size Statistics (analysable vesicles only)...")

    stats = (
        df_analysable.groupby('Category')['Refined Radius (um)']
        .agg(['mean', 'std', 'sem', 'count'])
        .rename(columns={
            'mean':  'Mean_GUV_radius_um',
            'std':   'Std_Dev_um',
            'sem':   'SEM_um',
            'count': 'N_Vesicles',
        })
        .reset_index()
    )

    path = os.path.join(output_dir, "Size_Statistics_Summary.csv")
    stats.to_csv(path, index=False)
    print(f"  -> Size statistics saved: {path}")


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