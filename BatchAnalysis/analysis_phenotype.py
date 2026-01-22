# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import mannwhitneyu 
import plotting

def categorize_vesicles(df):
    """Classifies vesicles based on the decision tree."""
    
    # Threshold Constants
    NO_LOC_LIMIT = 0.4          # Below this is Empty/Lumenal
    FUZZY_LOC_LIMIT = 0.5       # Below this (but >0.4) is Fuzzy
    PATCHY_LOC_LIMIT = 0.65     # Below this (but >0.5) is Patchy
    
    RHO_LIMIT = 8               # Distinguishes Empty vs Lumenal Actin
    THICKNESS_LIMIT = 0.8       # Used to exclude out-of-focus Fuzzy vesicles
    UNIFORMITY_LIMIT = 0.4      # New check: separates true Uniform from Patchy
    
    def classify(row):
        loc = row['A localization']
        rho = row['rho_actin']
        thick = row['t_cortex']
        cv = row['uniformity']
        
        # 1. Base Exclusion
        if pd.isna(loc) or pd.isna(rho) or pd.isna(thick):
            return "Excluded"
        
        # 2. Low Localization Branch (loc < 0.4)
        if loc < NO_LOC_LIMIT:
            if rho < RHO_LIMIT: return "Empty"
            else: return "Lumenal Actin"
            
        # 3. Fuzzy Branch (0.4 <= loc < 0.5)
        if loc < FUZZY_LOC_LIMIT:
            # EXCLUSION RULE: If fuzzy AND thick (>0.8), it's out-of-focus -> Exclude
            if thick > THICKNESS_LIMIT:
                return "Excluded"
            else:
                return "Fuzzy"
            
        # 4. Cortex Branch (loc >= 0.5)
        
        # A. Mid Localization (0.5 - 0.65) -> Always Patchy
        if loc < PATCHY_LOC_LIMIT:
            return "Patchy"
            
        # B. High Localization (> 0.65) -> Check Uniformity
        else:
            if pd.isna(cv): return "Patchy" # Fallback if cv is missing
            
            if cv < UNIFORMITY_LIMIT:
                return "Uniform"
            else:
                return "Patchy" # High loc, but not uniform enough

    df['Phenotype_Category'] = df.apply(classify, axis=1)
    return df

def save_phenotype_statistics(df, output_dir):
    """Calculates summary stats for phenotypes."""
    print("      -> Calculating Phenotype Statistics...")
    clean_df = df[df['Phenotype_Category'] != 'Excluded'].copy()
    target_cols = ['A localization', 't_cortex', 'rho_actin', 'uniformity']
    stats = clean_df.groupby('Phenotype_Category')[target_cols].agg(['mean', 'std', 'sem', 'count'])
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    path = os.path.join(output_dir, "Phenotype_Statistics_Summary.csv")
    stats.to_csv(path, index=False)

def add_stat_annotation(ax, df, x_col, y_col, group1, group2, order, level=0):
    """
    Draws a bracket and significance stars.
    """
    # 1. Get Data for the specific pair
    data1 = df[df[x_col] == group1][y_col].dropna()
    data2 = df[df[x_col] == group2][y_col].dropna()
    
    if len(data1) < 3 or len(data2) < 3: return 

    # 2. Run Test
    stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
    if p < 0.001: sig = "***"
    elif p < 0.01: sig = "**"
    elif p < 0.05: sig = "*"
    else: sig = "ns"

    # 3. Find Positions
    try:
        x1 = order.index(group1)
        x2 = order.index(group2)
    except ValueError: return 
    
    # 4. Determine Height
    start, end = min(x1, x2), max(x1, x2)
    categories_under_bracket = [order[i] for i in range(start, end+1)]
    
    subset = df[df[x_col].isin(categories_under_bracket)][y_col]
    if subset.empty: y_max = 0
    else: y_max = subset.max()
    
    y_range = df[y_col].max() - df[y_col].min()
    if y_range == 0: y_range = 1
    
    base_clearance = y_range * 0.05
    step = y_range * 0.10
    
    y_h = y_max + base_clearance + (level * step)
    y_text = y_h + (y_range * 0.02)
    
    # 5. Draw
    ax.plot([x1, x1, x2, x2], [y_h - (y_range*0.02), y_h, y_h, y_h - (y_range*0.02)], lw=1.5, c='black')
    ax.text((x1+x2)*.5, y_text, sig, ha='center', va='bottom', color='black', fontsize=11, fontweight='bold')

def generate_phenotype_panel(df, output_dir):
    """Creates 2x2 Panel with STACKED statistical brackets."""
    print("      -> Creating 2x2 Phenotype Panel (with stacked stats)...")
    
    plot_df = df[df['Phenotype_Category'] != 'Excluded'].copy()
    
    # UPDATE: Removed "Lumen-extended" from order
    order = ["Patchy", "Uniform", "Fuzzy", "Lumenal Actin", "Empty"]
    existing_order = [c for c in order if c in plot_df['Phenotype_Category'].unique()]
    
    params = [
        ('A localization', 'A Localization by Phenotype'),
        ('t_cortex', 't_cortex by Phenotype'),
        ('rho_actin', 'Rho_actin by Phenotype'),
        ('uniformity', 'uniformity by Phenotype')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    axes = axes.flatten() 

    for i, (col, title) in enumerate(params):
        ax = axes[i]
        if col in plot_df.columns:
            sns.boxplot(
                data=plot_df, x='Phenotype_Category', y=col, hue='Phenotype_Category',
                order=existing_order, legend=False, ax=ax, palette='tab10', showfliers=True 
            )
            
            # 1. Compare the two main cortex types
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, 
                              "Patchy", "Uniform", existing_order, level=0)
            
            # 2. Compare Transition (Fuzzy) vs Stable (Uniform)
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, 
                              "Fuzzy", "Uniform", existing_order, level=1)
            
            # 3. NEW: Compare Fuzzy vs Patchy (Level 2 - High/Spanning)
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, 
                              "Fuzzy", "Patchy", existing_order, level=2)
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Phenotype', fontsize=11)
            ax.set_ylabel(col, fontsize=11)
            ax.tick_params(axis='x', rotation=45) 
            ax.set_ylim(top=ax.get_ylim()[1] * 1.25) 
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, f"Column '{col}' not found", ha='center')

    plt.tight_layout()
    plotting.save_plot("Phenotype_Characteristics_Panel_2x2.png", output_dir)

def plot_5d_scatter(df, output_dir):
    """Generates 5D Scatter Plot."""
    print("      -> Creating 5D Spatial Map...")
    # Safety: Rename radius if needed
    if 'Refined Radius (um)' not in df.columns and 'radius' in df.columns:
        df['Refined Radius (um)'] = df['radius']
        
    plot_df = df[df['t_cortex'] > 0.05].copy()
    if plot_df.empty: return

    plt.figure(figsize=(12, 9), dpi=125)
    sns.scatterplot(
        data=plot_df, x='t_cortex', y='rho_actin', hue='A localization', 
        size='Refined Radius (um)', style='Phenotype_Category',
        sizes=(30, 400), palette='viridis', alpha=0.75, edgecolor='black'
    )
    plt.title('5D Cortex Map: Thickness vs Density\n(Color=Loc, Size=Radius, Shape=Phenotype)', fontsize=15)
    plt.xlabel('Cortex Thickness (µm)', fontsize=12)
    plt.ylabel('Actin Density (a.u.)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Properties")
    plotting.save_plot("Cortex_5D_Bubble_Map.png", output_dir)

def plot_pairgrid(df, output_dir):
    """Generates Pair Grid."""
    print("      -> Creating Correlation Matrix...")
    cols = ['t_cortex', 'rho_actin', 'A localization', 'uniformity', 'Phenotype_Category']
    plot_df = df[df['t_cortex'] > 0.05][cols].copy()
    if plot_df.empty: return

    g = sns.pairplot(plot_df, hue='Phenotype_Category', palette='tab10', corner=True, plot_kws={'alpha': 0.6, 's': 40})
    g.fig.suptitle("Multi-Variable Correlation Matrix", y=1.02, fontsize=16)
    path = os.path.join(output_dir, "Correlation_Matrix_PairPlot.png")
    g.savefig(path, bbox_inches='tight')
    plt.close()

def run_phenotype_analysis(df, output_dir):
    """Main runner."""
    print("\n--- Running Phenotype & Cortex Analysis ---")

    df = categorize_vesicles(df)
    
    # Save Log
    log_cols = ['Date', 'Name', 'Vesicle id', 'Phenotype_Category', 'A localization', 't_cortex', 'rho_actin', 'uniformity']
    existing_cols = [c for c in log_cols if c in df.columns]
    log_path = os.path.join(output_dir, "Phenotype_Categorization_Log.csv")
    df[existing_cols].to_csv(log_path, index=False)
    
    plotting.plot_phenotype_composition(df, 'Category', 'Phenotype_Category', output_dir, 'Phenotype_Composition_Stacked.png')
    
    # Run Updated Panel with Stats
    generate_phenotype_panel(df, output_dir)
    plot_5d_scatter(df, output_dir)
    plot_pairgrid(df, output_dir)
    save_phenotype_statistics(df, output_dir)