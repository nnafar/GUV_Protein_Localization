# -*- coding: utf-8 -*-
"""
BATCH ANALYSIS MODULE
Detects and visualises batch-to-batch variability within each condition.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal   # non-parametric test for >2 groups
import plotting


# =============================================================================
# WHICH METRICS TO COMPARE ACROSS BATCHES
# =============================================================================

# These metrics are meaningful for EVERY condition (controls and experimental).
# 'Refined Radius (um)' tells you vesicle size — should be stable batch-to-batch.
# 'Solidity' tells you shape quality — also should be stable.
METRICS_ALL_CONDITIONS = [
    'Refined Radius (um)',
    'Solidity',
]

# These extra metrics only exist when actin data is present.
METRICS_ACTIN_CONDITIONS = [
    'A localization',
    'Gini_Index',
    'ISM',
    't_cortex',
]

# The set of conditions that have actin channels.
CONDITIONS_WITH_ACTIN = {'BranchedCortex', 'LinearCortex', 'Factin'}

# Human-readable axis labels for the metrics.
# (Keeps the axis tidy — shorter than the raw column name.)
METRIC_LABELS = {
    'Refined Radius (um)':  'GUV radius (µm)',
    'Solidity':             'Solidity',
    'A localization':       'Localization Score',
    'Gini_Index':           'Gini Index',
    'ISM':                  'ISM',
    't_cortex':             'Cortex Thickness (µm)',
}


# =============================================================================
# HELPER — DECIDE WHICH METRICS TO USE FOR A CONDITION
# =============================================================================

def _get_metrics_for_condition(condition, df_columns):
    """
    Returns the list of metric names relevant for the given condition,
    filtered to only those that actually exist as columns in the data.

    WHY FILTER AGAINST df_columns?
    If a condition (like Empty) has no actin, the actin columns simply
    won't exist — asking for them would crash the code. This function
    prevents that crash by only returning columns that are present.

    Parameters
    ----------
    condition  : str  — e.g. 'BranchedCortex'
    df_columns : list — column names that exist in the DataFrame

    Returns
    -------
    metrics : list of str
    """
    # Start with the universal metrics
    metrics = list(METRICS_ALL_CONDITIONS)

    # Add actin metrics only for conditions that have an actin channel
    if condition in CONDITIONS_WITH_ACTIN:
        metrics += METRICS_ACTIN_CONDITIONS

    # Keep only columns that actually exist — avoids KeyError crashes
    metrics = [m for m in metrics if m in df_columns]

    return metrics


# =============================================================================
# STEP 1 — BATCH STATISTICS CSV
# =============================================================================

def compute_batch_statistics(df, output_dir):
    """
    Computes mean, standard deviation, median, and N for each metric,
    grouped by (Condition, Batch_ID).

    THINK OF IT LIKE:
    A school report card where each row is one class (batch),
    each column is one subject (metric), and the value is the class average.

    Parameters
    ----------
    df         : pandas DataFrame  — must have 'Category' and 'Batch_ID'
    output_dir : str               — folder to save the CSV

    Returns
    -------
    stats_df : pandas DataFrame with the computed statistics
    """
    print("  -> Computing batch statistics...")

    # Combine both metric lists, then filter to those present in this dataset
    all_metrics = METRICS_ALL_CONDITIONS + METRICS_ACTIN_CONDITIONS
    all_metrics = [m for m in all_metrics if m in df.columns]

    rows = []

    # groupby() splits the DataFrame into sub-tables, one per unique
    # (Category, Batch_ID) pair, then loops through them.
    for (condition, batch_id), group in df.groupby(['Category', 'Batch_ID']):

        row = {
            'Condition': condition,
            'Batch_ID':  batch_id,
            'N_Vesicles': len(group),   # how many vesicles in this batch
        }

        for metric in all_metrics:
            if metric not in group.columns:
                continue
            values = group[metric].dropna()   # ignore missing (NaN) values

            if len(values) == 0:
                # No valid data for this metric in this batch
                row[f'{metric}_mean']   = np.nan
                row[f'{metric}_std']    = np.nan
                row[f'{metric}_median'] = np.nan
                row[f'{metric}_N']      = 0
            else:
                row[f'{metric}_mean']   = round(float(values.mean()),   4)
                row[f'{metric}_std']    = round(float(values.std()),    4)
                row[f'{metric}_median'] = round(float(values.median()), 4)
                row[f'{metric}_N']      = int(len(values))

        rows.append(row)

    stats_df = pd.DataFrame(rows).sort_values(['Condition', 'Batch_ID'])

    path = os.path.join(output_dir, "Batch_Statistics_Summary.csv")
    stats_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    return stats_df


# =============================================================================
# STEP 2 — KRUSKAL-WALLIS CONSISTENCY TESTS
# =============================================================================

def run_batch_significance_tests(df, output_dir):
    """
    Tests whether batches within each condition are statistically
    consistent using the Kruskal-Wallis test.

    WHAT IS THE KRUSKAL-WALLIS TEST?
    ---------------------------------
    It is a non-parametric test (does not assume your data is normally
    distributed) that asks: "Are all these groups drawn from the same
    distribution, or is at least one group different?"

    - p-value ≥ 0.05  →  batches are CONSISTENT  (no significant difference)
    - p-value < 0.05  →  batches are INCONSISTENT  (something changed)

    ANALOGY:
    You weigh 5 bags of flour from 5 different factory runs.
    The test asks: "Is one delivery systematically heavier than the others?"

    IMPORTANT CAVEAT:
    A significant result (p < 0.05) does NOT mean the experiment is ruined.
    It just means you should LOOK at the batch plot and decide whether the
    difference is scientifically meaningful.

    Parameters
    ----------
    df         : pandas DataFrame
    output_dir : str

    Returns
    -------
    list of result dicts (also saved to CSV)
    """
    print("  -> Running batch consistency tests (Kruskal-Wallis)...")

    results = []

    for condition in df['Category'].unique():
        cond_df  = df[df['Category'] == condition]
        batches  = cond_df['Batch_ID'].unique()

        # Need at least 2 batches to make a comparison
        if len(batches) < 2:
            print(f"      -> Skipping {condition}: only 1 batch found")
            continue

        metrics = _get_metrics_for_condition(condition, df.columns.tolist())

        for metric in metrics:

            # Collect the values for each batch as a separate array
            # (kruskal() needs them as separate arguments, not one big list)
            groups = []
            for batch in batches:
                vals = (cond_df
                        .loc[cond_df['Batch_ID'] == batch, metric]
                        .dropna()
                        .values)
                # Only include a batch if it has at least 3 data points
                if len(vals) >= 3:
                    groups.append(vals)

            # Can't run the test with fewer than 2 valid groups
            if len(groups) < 2:
                continue

            try:
                # The * unpacks the list: kruskal(group1, group2, group3, ...)
                stat, p_val = kruskal(*groups)

                # Translate p-value into a plain English interpretation
                if p_val < 0.001:
                    interp = "Highly inconsistent ***"
                elif p_val < 0.01:
                    interp = "Inconsistent **"
                elif p_val < 0.05:
                    interp = "Borderline inconsistent *"
                else:
                    interp = "Consistent (no significant difference)"

                results.append({
                    'Condition':      condition,
                    'Metric':         metric,
                    'N_Batches':      len(groups),
                    'KW_statistic':   round(float(stat),  4),
                    'p_value':        round(float(p_val), 6),
                    'Interpretation': interp,
                })

            except Exception as e:
                print(f"      ! Test failed ({condition}, {metric}): {e}")

    if results:
        results_df = (pd.DataFrame(results)
                      .sort_values(['Condition', 'p_value']))

        path = os.path.join(output_dir, "Batch_Consistency_Tests.csv")
        results_df.to_csv(path, index=False)
        print(f"      -> Saved: {path}")

        # Print a readable summary in the console
        # ✓ = consistent  |  ⚠ = inconsistent (may need attention)
        print("\n      Batch consistency summary:")
        for _, row in results_df.iterrows():
            flag = "  ⚠" if row['p_value'] < 0.05 else "  ✓"
            print(f"{flag}  {row['Condition']:>15}  |  "
                  f"{row['Metric']:>22}  |  "
                  f"p = {row['p_value']:.4f}  |  "
                  f"{row['Interpretation']}")
        print()

    else:
        print("      -> Not enough batches for statistical testing "
              "(need ≥ 2 batches per condition).")

    return results


# =============================================================================
# STEP 3 — SHORTEN BATCH LABELS FOR PLOT AXES
# =============================================================================

def _shorten_batch_label(batch_id, condition):
    """
    Converts a long batch folder name into a short axis label.

    Example:
        '260208_BranchedCortex_1'  +  'BranchedCortex'
        → '260208 #1'

    This keeps the x-axis readable even when batch names are long.

    Parameters
    ----------
    batch_id  : str — e.g. '260208_BranchedCortex_1'
    condition : str — e.g. 'BranchedCortex'

    Returns
    -------
    str — short label, e.g. '260208 #1'
    """
    # Remove the condition name from the batch ID (case-insensitive)
    short = batch_id.replace(condition, '').replace('__', '_').strip('_')

    # Split by '_' and try to find a date-like part and a run number
    parts = [p for p in short.split('_') if p]

    if len(parts) >= 2:
        # First part is typically the date, last part the run number
        return f"{parts[0]} #{parts[-1]}"
    elif len(parts) == 1:
        return parts[0]
    else:
        return batch_id   # fall back to the full name if parsing fails


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_batch_analysis(df, output_dir):
    """
    Main function that runs ALL batch analyses in sequence.

    Call this from master_pipeline.py after the other analysis modules.

    Steps:
      1. Compute and save per-batch statistics CSV
      2. Run Kruskal-Wallis consistency tests, save results CSV
      3. Generate one per-condition batch variability plot
      4. Generate one overview plot showing all conditions

    Parameters
    ----------
    df         : pandas DataFrame — the master dataset from file_handling.py
    output_dir : str              — the Batch_Analysis_Results folder path
    """
    print("\n--- Running Batch-to-Batch Variability Analysis ---")

    # Safety check — tell the user clearly if something is missing
    if 'Batch_ID' not in df.columns:
        print("  ! 'Batch_ID' column not found.")
        print("    Make sure you are using the current version of file_handling.py.")
        return

    if 'Category' not in df.columns:
        print("  ! 'Category' column not found.")
        return

    # Create a dedicated sub-folder so batch outputs don't clutter the main folder
    batch_output_dir = os.path.join(output_dir, "Batch_Analysis")
    os.makedirs(batch_output_dir, exist_ok=True)

    # ── Step 1: Statistics CSV ───────────────────────────────────────────────
    compute_batch_statistics(df, batch_output_dir)

    # ── Step 2: Significance tests ───────────────────────────────────────────
    run_batch_significance_tests(df, batch_output_dir)

    # ── Add short batch labels to the full dataset ───────────────────────────
    # These labels (e.g. '260208 #1') are used on the x-axes of every plot.
    # We do this once here and pass the labelled DataFrame into every plot
    # function, so each function doesn't have to repeat the logic.
    plotting.set_paper_style()

    df_labelled = df.copy()
    df_labelled['Batch_Label'] = df_labelled.apply(
        lambda row: _shorten_batch_label(row['Batch_ID'], row['Category']),
        axis=1
    )

    # ── Step 3: Size distribution — all conditions, one figure ───────────────
    # Shows how vesicle RADIUS varies across batches within each condition.
    # A good experiment has boxes at similar heights in every batch.
    plotting.plot_batch_size_distribution(df_labelled, batch_output_dir)

    # ── Step 4: Phenotype composition — all conditions, one figure ────────────
    # Shows the PERCENTAGE of each phenotype per batch.
    # Requires that analysis_phenotype.run_phenotype_analysis() was called
    # before run_batch_analysis() — master_pipeline.py ensures this ordering.
    if 'Phenotype_Category' in df_labelled.columns:
        plotting.plot_batch_phenotype_composition(df_labelled, batch_output_dir)
    else:
        print("  -> Skipping phenotype composition plot "
              "('Phenotype_Category' column not found — "
              "check that phenotype analysis ran before batch analysis).")

    # ── Step 5: Actin metrics — t_cortex, localization, ISM, Gini ────────────
    # Only plotted for conditions that have an actin channel.
    # Empty is skipped automatically inside the function.
    plotting.plot_batch_actin_metrics(df_labelled, batch_output_dir)

    # ── Step 6: Cortex-forming-only companion ─────────────────────────────────
    # Same as Step 5 but EMPTY + LUMENAL GUVs are excluded first, so the
    # dashed median line and violin shapes reflect only GUVs that actually
    # formed a membrane-associated structure.
    #
    # This call lives here (not in analysis_statistics.py) because it needs
    # 'Batch_Label', which is created above in this function.
    if 'Phenotype_Category' in df_labelled.columns:
        plotting.plot_batch_actin_metrics_cortex_only(df_labelled, batch_output_dir)
    else:
        print("  -> Skipping cortex-only batch actin plot "
              "('Phenotype_Category' not found — run phenotype analysis first).")

    print("  -> Batch Analysis Complete.\n")

# NOTE: plot_batch_actin_metrics is called from run_batch_analysis above.
# The call is appended below by patching the file end.