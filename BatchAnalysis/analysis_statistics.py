# -*- coding: utf-8 -*-
"""
STATISTICS MODULE
Performs hypothesis testing and effect size calculation.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import plotting

def cliffs_delta(lst1, lst2):
    """
    Calculates Cliff's Delta effect size for two lists of values.
    Returns value between 0 and 1.
    """
    lst1 = np.sort(lst1)
    lst2 = np.sort(lst2)
    m, n = len(lst1), len(lst2)
    if m == 0 or n == 0:
        return 0
        
    j = 0
    k = 0
    more = 0
    less = 0
    
    for x in lst1:
        while j < n and lst2[j] < x:
            j += 1
        while k < n and lst2[k] <= x:
            k += 1
        more += j
        less += (n - k)
        
    delta = (more - less) / (m * n)
    return abs(delta)

def interpret_effect(d):
    """Interprets Cliff's Delta magnitude."""
    if d < 0.147: return "Negligible"
    if d < 0.33: return "Small"
    if d < 0.474: return "Medium"
    return "Large"

def get_significance_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def run_phenotype_per_category_stats(df, output_dir):
    """
    Drills down into each Category (Branched/Linear) and compares 
    Phenotypes (Sparse vs Patchy, Patchy vs Continuous).
    """
    print("  -> Running Within-Category Phenotype Statistics...")
    
    results = []
    # ADDED: Radial Kurtosis
    metrics = ['t_cortex', 'A localization', 'ISM', 'Gini_Index', 'Radial_Kurtosis']
    
    pheno_pairs = [
        ('SPARSE', 'PATCHY'),
        ('PATCHY', 'CONTINUOUS'),
        ('SPARSE', 'CONTINUOUS')
    ]
    
    categories = df['Category'].unique()

    for cat in categories:
        cat_df = df[df['Category'] == cat]
        
        for metric in metrics:
            valid_df = cat_df.dropna(subset=[metric])
            
            for p1, p2 in pheno_pairs:
                group1 = valid_df[valid_df['Phenotype_Category'] == p1][metric].values
                group2 = valid_df[valid_df['Phenotype_Category'] == p2][metric].values
                
                if len(group1) > 2 and len(group2) > 2:
                    try:
                        stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                        d_val = cliffs_delta(group1, group2)
                        
                        results.append({
                            'Category': cat,
                            'Metric': metric,
                            'Comparison': f"{p1} vs {p2}",
                            'p-value': p_val,
                            'Significance': get_significance_stars(p_val),
                            'Effect_Size': d_val,
                            'Effect_Interp': interpret_effect(d_val),
                            'N1': len(group1),
                            'N2': len(group2)
                        })
                    except Exception as e:
                        print(f"    ! Error comparing {p1}-{p2} in {cat}: {e}")

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['Category', 'Metric'])
        
        filename = "Statistical_Report_Phenotypes_Per_Category.csv"
        path = os.path.join(output_dir, filename)
        results_df.to_csv(path, index=False)
        print(f"      -> Detailed Stats saved: {filename}")
    else:
        print("      -> No valid comparisons found (insufficient data).")

def run_statistics(df, output_dir):
    print("\n--- Running Statistical Analysis ---")
    
    # 1. Standard Correlation Matrix
    # We only correlate numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr(method='spearman')
        plotting.plot_correlation_heatmap(corr_matrix, output_dir, "_Global")
    
    # 2. Run the new "Within-Category" Analysis
    run_phenotype_per_category_stats(df, output_dir)