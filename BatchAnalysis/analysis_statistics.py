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

def run_spearman_correlation(df, output_dir):
    """Calculates and plots Spearman Correlation Matrix."""
    print("  -> Calculating Spearman Correlations...")
    
    # Ensure radius column exists
    if 'Refined Radius (um)' not in df.columns and 'radius' in df.columns:
        df['Refined Radius (um)'] = df['radius']
        
    target_cols = ['Refined Radius (um)', 't_cortex', 'rho_actin', 'A localization', 'uniformity']
    valid_cols = [c for c in target_cols if c in df.columns]
    
    # Filter for formed cortices (>0.05) to avoid noise from empty vesicles
    corr_df = df[df['t_cortex'] > 0.05][valid_cols].copy()
    
    if corr_df.empty or len(valid_cols) < 2:
        print("    ! Not enough data for correlation.")
        return

    # Calculate Spearman (Rank-based)
    corr_matrix = corr_df.corr(method='spearman')
    
    # Save CSV
    path = os.path.join(output_dir, "Spearman_Correlation_Matrix.csv")
    corr_matrix.to_csv(path)
    print(f"    -> Matrix saved: {path}")
    
    # Plot Heatmap
    plotting.plot_correlation_heatmap(corr_matrix, output_dir)


def run_statistics(df, output_dir):
    print("\n--- Running Statistical Analysis (Mann-Whitney U + Spearman) ---")
    
    # 1. Run Hypothesis Tests (Group Differences)
    results = []
    cortex_df = df[df['t_cortex'] > 0.05].copy()

    # Define the comparisons you care about
    comparisons = [
        # Experimental
        ('Category', 'BranchedCortex', 'LinearCortex'),
        
        # Main Pathway (Keep these)
        ('Phenotype_Category', 'Sparse', 'Uniform'),
        ('Phenotype_Category', 'Patchy', 'Uniform'),
        ('Phenotype_Category', 'Sparse', 'Patchy'),
        ('Phenotype_Category', 'Empty', 'Lumenal Actin'), # Validates detection limit
        ('Phenotype_Category', 'Lumenal Actin', 'Sparse')  # Validates Sparse structure
    ]
    
    metrics = ['t_cortex', 'rho_actin', 'A localization', 'uniformity']

    for col, g1, g2 in comparisons:
        # Check if groups actually exist in the data
        if g1 in df[col].values and g2 in df[col].values:
            print(f"  -> Testing {g1} vs {g2}...")
            for metric in metrics:
                res = perform_mann_whitney(cortex_df, col, metric, g1, g2)
                if res: results.append(res)

    if results:
        stats_df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, "Statistical_Report_Full.csv")
        stats_df.to_csv(output_path, index=False)
        print(f"  -> Report saved: {output_path}")
    else:
        print("  ! No valid comparisons found.")
        
    # 2. Run Correlation Analysis (Relationships)
    run_spearman_correlation(df, output_dir)