# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import plotting

def cliffs_delta(lst1, lst2):
    """
    Calculates Cliff's Delta effect size for two lists of values.
    Returns value between -1 and 1.
    0.0 = no difference.
    1.0 = complete separation.
    """
    m, n = len(lst1), len(lst2)
    lst1 = np.sort(lst1)
    lst2 = np.sort(lst2)
    j = 0
    k = 0
    more = 0
    less = 0
    
    # Efficient calculation of dominance
    for x in lst1:
        while j < n and lst2[j] < x:
            j += 1
        while k < n and lst2[k] <= x:
            k += 1
        more += j
        less += (n - k)
        
    delta = (more - less) / (m * n)
    return abs(delta) # Return absolute magnitude

def interpret_effect(d):
    """Interprets Cliff's Delta magnitude."""
    if d < 0.147: return "Negligible"
    if d < 0.33: return "Small"
    if d < 0.474: return "Medium"
    return "Large"

def perform_mann_whitney(df, group_col, value_col, group1, group2):
    """
    Runs a Mann-Whitney U test AND calculates Cliff's Delta.
    """
    # Extract data
    data1 = df[df[group_col] == group1][value_col].dropna().values
    data2 = df[df[group_col] == group2][value_col].dropna().values

    if len(data1) < 3 or len(data2) < 3:
        return None

    # 1. Run Hypothesis Test (p-value)
    stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
    
    # 2. Run Effect Size (Magnitude of difference)
    delta = cliffs_delta(data1, data2)
    effect_str = interpret_effect(delta)

    # Determine stars
    if p < 0.001: sig = "***"
    elif p < 0.01: sig = "**"
    elif p < 0.05: sig = "*"
    else: sig = "ns"

    return {
        "Comparison": f"{group1} vs {group2}",
        "Metric": value_col,
        "N1": len(data1),
        "N2": len(data2),
        "p-value": f"{p:.5e}",
        "Significance": sig,
        "Effect Size (Cliff's d)": f"{delta:.2f}",
        "Effect Strength": effect_str
    }

def run_spearman_correlation(df, output_dir, filename_suffix=""):
    """Calculates and plots Spearman Correlation Matrix."""
    if 'Refined Radius (um)' not in df.columns and 'radius' in df.columns:
        df['Refined Radius (um)'] = df['radius']
        
    target_cols = ['Refined Radius (um)', 't_cortex', 'rho_actin', 'A localization', 'uniformity']
    valid_cols = [c for c in target_cols if c in df.columns]
    
    corr_df = df[df['t_cortex'] > 0.05][valid_cols].copy()
    if corr_df.empty or len(valid_cols) < 2: return

    corr_matrix = corr_df.corr(method='spearman')
    path = os.path.join(output_dir, f"Spearman_Correlation_Matrix{filename_suffix}.csv")
    corr_matrix.to_csv(path)
    
    # Pass suffix to plotting function
    plotting.plot_correlation_heatmap(corr_matrix, output_dir, filename_suffix)


def run_statistics(df, output_dir):
    print("\n--- Running Statistical Analysis ---")
    
    # 1. Global Hypothesis Tests
    results = []
    cortex_df = df[df['t_cortex'] > 0.05].copy()
    comparisons = [
        ('Category', 'BranchedCortex', 'LinearCortex'),
        ('Phenotype_Category', 'Sparse', 'Uniform'),
        ('Phenotype_Category', 'Patchy', 'Uniform'),
    ]
    metrics = ['t_cortex', 'rho_actin', 'A localization', 'uniformity']

    for col, g1, g2 in comparisons:
        if g1 in df[col].values and g2 in df[col].values:
            for metric in metrics:
                res = perform_mann_whitney(cortex_df, col, metric, g1, g2)
                if res: results.append(res)
    if results:
        pd.DataFrame(results).to_csv(os.path.join(output_dir, "Statistical_Report_Full.csv"), index=False)

    # 2. SPLIT CORRELATIONS: Loop through categories
    unique_cats = df['Category'].unique()
    print(f"  -> Generating Heatmaps for: {unique_cats}")
    
    # A. Global Heatmap
    run_spearman_correlation(df, output_dir, filename_suffix="_Global")
    
    # B. Per-Category Heatmap
    for cat in unique_cats:
        cat_df = df[df['Category'] == cat].copy()
        run_spearman_correlation(cat_df, output_dir, filename_suffix=f"_{cat}")