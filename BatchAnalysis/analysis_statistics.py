# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

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

def run_statistics(df, output_dir):
    print("\n--- Running Statistical Analysis (Mann-Whitney U + Effect Size) ---")
    
    results = []
    cortex_df = df[df['t_cortex'] > 0.05].copy()

    # Define the comparisons you care about
    # Format: (Column, Group1, Group2)
    comparisons = [
        # Experimental
        ('Category', 'BranchedCortex', 'LinearCortex'),
        
        # Phenotypic - Transition vs Stable
        ('Phenotype_Category', 'Fuzzy', 'Uniform'),
        
        # Phenotypic - Stability (Patchy vs Uniform)
        ('Phenotype_Category', 'Patchy', 'Uniform'),

        # NEW: Phenotypic - Transition vs Formed Patchy
        ('Phenotype_Category', 'Fuzzy', 'Patchy')
    ]
    
    metrics = ['t_cortex', 'rho_actin', 'A localization', 'uniformity']

    for col, g1, g2 in comparisons:
        # Check if groups actually exist in the data
        if g1 in df[col].values and g2 in df[col].values:
            print(f"  -> Testing {g1} vs {g2}...")
            for metric in metrics:
                # Skip uniformity if not relevant (e.g. for Fuzzy vs Thick)
                #if metric == 'uniformity' and 'Thick' in g1: continue 
                
                res = perform_mann_whitney(cortex_df, col, metric, g1, g2)
                if res: results.append(res)

    if results:
        stats_df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, "Statistical_Report_Full.csv")
        stats_df.to_csv(output_path, index=False)
        
        print(f"  -> Report saved: {output_path}")
        # Print a clean summary to console
        print("\n" + stats_df[['Comparison', 'Metric', 'p-value', 'Significance', 'Effect Strength']].to_string(index=False))
    else:
        print("  ! No valid comparisons found.")