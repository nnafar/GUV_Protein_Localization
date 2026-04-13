# -*- coding: utf-8 -*-
"""
STATISTICS MODULE
Performs hypothesis testing and effect size calculation.

CHANGES FROM PREVIOUS VERSION:
-------------------------------
1. Within-condition statistics are now CONDITION-AWARE.
   Each condition has a defined set of phenotype comparisons that make sense
   for it. For example, comparing SPARSE vs PATCHY only applies to
   BranchedCortex and LinearCortex — not to Factin (which only has SHELL/LUMENAL)
   or Empty (no phenotype variation).

2. Graceful skipping: if a comparison group has too few data points (< 3),
   it is silently skipped instead of crashing.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import plotting


# =============================================================================
# 1. EFFECT SIZE HELPERS
# =============================================================================

def cliffs_delta(lst1, lst2):
    """
    Calculates Cliff's Delta, a measure of how different two groups are.

    Think of it like asking: "If I pick one value from group A and one from
    group B at random, how often is the one from A larger?"
    - Delta near 0   → groups are very similar
    - Delta near 1   → groups are very different

    Returns a value between 0 and 1 (we use the absolute value).
    """
    lst1 = np.sort(lst1)
    lst2 = np.sort(lst2)
    m, n = len(lst1), len(lst2)
    if m == 0 or n == 0:
        return 0

    j = k = more = less = 0
    for x in lst1:
        while j < n and lst2[j] < x:
            j += 1
        while k < n and lst2[k] <= x:
            k += 1
        more += j
        less += (n - k)

    return abs((more - less) / (m * n))


def interpret_effect(d):
    """Returns a human-readable label for a Cliff's Delta value."""
    if d < 0.147: return "Negligible"
    if d < 0.330: return "Small"
    if d < 0.474: return "Medium"
    return "Large"


def get_significance_stars(p):
    """Converts a p-value to the standard star notation."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# =============================================================================
# 2. CONDITION-SPECIFIC PHENOTYPE COMPARISON PAIRS
# =============================================================================

# This dictionary defines WHICH phenotype pairs to compare for each condition.
# If a condition isn't listed here, no within-condition stats are run for it.
#
# Format: 'ConditionName': [ ('PhenotypeA', 'PhenotypeB'), ... ]
#
# Analogy: think of it as a list of match-ups in a sports tournament.
# You only schedule matches between teams that actually exist in that league.

CONDITION_PHENO_PAIRS = {
    'BranchedCortex': [
        ('SPARSE', 'PATCHY'),
        ('PATCHY', 'CONTINUOUS'),
        ('SPARSE', 'CONTINUOUS'),
    ],
    'LinearCortex': [
        # Only one comparison — PATCHY does not exist for LinearCortex.
        # Gini_Index shows no separation between any sub-groups, so the
        # SPARSE/PATCHY/CONTINUOUS scheme used for BranchedCortex does not
        # apply here.
        ('SPARSE', 'CONTINUOUS'),
    ],
    'Factin': [
        ('SHELL', 'LUMENAL'),
    ],
    # Empty is intentionally absent — no meaningful phenotype comparisons
}

# The metrics to test for each condition.
# Radial_Kurtosis has been removed: the cortical window contains only ~17-21
# data points (membrane peak width + padding), which is far below the ~50
# needed for reliable kurtosis estimates. It was also strongly collinear
# with ISM (r≈0.90) and Gini_Index (r≈0.80), adding no new information.
METRICS_TO_TEST = [
    't_cortex',
    'A localization',
    'ISM',
    'Gini_Index',
]


# =============================================================================
# 3. MAIN STATISTICS FUNCTIONS
# =============================================================================

def run_phenotype_per_category_stats(df, output_dir):
    """
    For each condition, compares phenotype groups on the key metrics using
    the Mann-Whitney U test and Cliff's Delta effect size.

    Results are saved to a single CSV file.
    """
    print("  -> Running Within-Condition Phenotype Statistics...")

    results = []

    for condition, pheno_pairs in CONDITION_PHENO_PAIRS.items():
        # Select only rows belonging to this condition
        cond_df = df[df['Category'] == condition]

        if cond_df.empty:
            print(f"      ! No data found for condition: {condition}")
            continue

        for metric in METRICS_TO_TEST:
            # Drop rows where this metric is missing
            valid_df = cond_df.dropna(subset=[metric])

            for p1, p2 in pheno_pairs:
                group1 = valid_df[valid_df['Phenotype_Category'] == p1][metric].values
                group2 = valid_df[valid_df['Phenotype_Category'] == p2][metric].values

                # Skip if either group is too small to test
                if len(group1) < 3 or len(group2) < 3:
                    continue

                try:
                    stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                    d_val = cliffs_delta(group1, group2)

                    results.append({
                        'Condition':     condition,
                        'Metric':        metric,
                        'Comparison':    f"{p1} vs {p2}",
                        'p-value':       round(p_val, 6),
                        'Significance':  get_significance_stars(p_val),
                        'Effect_Size':   round(d_val, 4),
                        'Effect_Interp': interpret_effect(d_val),
                        'N1':            len(group1),
                        'N2':            len(group2),
                    })

                except Exception as e:
                    print(f"      ! Error ({condition}, {metric}, {p1} vs {p2}): {e}")

    if results:
        results_df = pd.DataFrame(results).sort_values(by=['Condition', 'Metric'])
        path = os.path.join(output_dir, "Statistical_Report_Phenotypes_Per_Condition.csv")
        results_df.to_csv(path, index=False)
        print(f"      -> Statistics saved: {path}")
    else:
        print("      -> No valid comparisons found (insufficient data).")


def run_statistics(df, output_dir):
    """
    Runs all statistical analyses:
      1. Spearman correlation heatmap (overall)
      2. Per-condition Spearman correlation heatmaps
      3. Within-condition phenotype comparisons
    """
    print("\n--- Running Statistical Analysis ---")

    # 1. Global Spearman correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr(method='spearman')
        plotting.plot_correlation_heatmap(corr_matrix, output_dir, "_Global")

    # 2. Per-condition Spearman correlation matrices
    #    Use the canonical order defined in plotting so figures are consistent.
    condition_order = [c for c in plotting.CONDITION_ORDER
                       if c in df['Category'].unique()]
    for condition in condition_order:
        cond_df = df[df['Category'] == condition].select_dtypes(include=[np.number])
        if cond_df.empty:
            continue
        # Drop columns that are entirely NaN for this condition
        cond_df = cond_df.dropna(axis=1, how='all')
        if cond_df.shape[1] < 2:
            continue
        corr_matrix = cond_df.corr(method='spearman')
        label = plotting.CONDITION_LABELS.get(condition, condition)
        plotting.plot_correlation_heatmap(corr_matrix, output_dir,
                                          f"_{condition}",
                                          title=f"Spearman Correlation — {label}")
        print(f"  -> Correlation heatmap saved for: {label}")

    # 3. Within-condition phenotype comparisons
    run_phenotype_per_category_stats(df, output_dir)