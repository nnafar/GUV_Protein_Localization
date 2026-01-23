# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import mannwhitneyu 
import plotting

# =============================================================================
# 1. LOGIC FUNCTIONS
# =============================================================================

def categorize_vesicles(df):
    """Classifies vesicles based on the decision tree."""
    
    # Threshold Constants
    NO_LOC_LIMIT = 0.4          # Below this is Empty/Lumenal
    SPARSE_LOC_LIMIT = 0.5       # Below this (but >0.4) is Fuzzy
    PATCHY_LOC_LIMIT = 0.65     # Below this (but >0.5) is Patchy
    
    RHO_LIMIT = 2               # Distinguishes Empty vs Lumenal Actin
    THICKNESS_LIMIT = 1.6       # Used to exclude out-of-focus Sparse vesicles
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
            
        # 3. Sparse Branch (0.4 <= loc < 0.5)
        if loc < SPARSE_LOC_LIMIT:
            # EXCLUSION RULE: If Sparse AND thick (>1.2), it's out-of-focus -> Exclude
            if thick > THICKNESS_LIMIT:
                return "Excluded"
            else:
                return "Sparse"
            
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

# =============================================================================
# 2. PLOTTING FUNCTIONS
# =============================================================================

def generate_phenotype_panel(df, output_dir):
    print("      -> Creating 2x2 Phenotype Panel (MFA Style)...")
    plot_df = df[df['Phenotype_Category'] != 'Excluded'].copy()
    
    # Plotting order:     
    order = ["Empty", "Lumenal Actin", "Sparse", "Patchy", "Uniform"]
    existing_order = [c for c in order if c in plot_df['Phenotype_Category'].unique()]
    
    params = [
        ('A localization', 'A Localization'),
        ('t_cortex', 'Cortex Thickness'),
        ('rho_actin', 'Actin Density'),
        ('uniformity', 'Uniformity (CV)')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten() 

    for i, (col, title) in enumerate(params):
        ax = axes[i]
        if col in plot_df.columns:
            # PASS THE ORDER HERE
            plotting.plot_boxplot_comparison(
                data=plot_df, x_col='Phenotype_Category', y_col=col, 
                title=title, ylabel=col, ax=ax, order=existing_order
            )
            
            # --- STATISTICAL ANNOTATIONS ---
           
            # 1. Validation: Empty vs Lumenal (Level 0)
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, "Empty", "Lumenal Actin", existing_order, level=0)
            
            # 2. Validation: Lumenal vs Sparse (Level 1)
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, "Lumenal Actin", "Sparse", existing_order, level=1)
            
            # 3. Comparison: Sparse vs Patchy (Level 0)
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, "Sparse", "Patchy", existing_order, level=0)
            
            # 4. Comparison: Patchy vs Uniform (Level 0)
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, "Patchy", "Uniform", existing_order, level=0)

            # 5. Comparison: Sparse vs Uniform (Level 2 - Higher up)
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, "Sparse", "Uniform", existing_order, level=2)
            
            # Clean up X-axis
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('') 
            ax.set_ylim(top=ax.get_ylim()[1] * 1.4) # Increase headroom for stacked brackets
        else:
            ax.text(0.5, 0.5, "Data Not Found", ha='center')

    plt.tight_layout()
    plotting.save_plot("Phenotype_Characteristics_Panel_2x2.png", output_dir)


def plot_5d_scatter(df, output_dir):
    """
    OPTION 1: Clean Phenotype Map (Color = Phenotype).
    - X vs Y shows the decision logic.
    - Color shows the PHENOTYPE (so you know what is what).
    - Size shows Radius.
    """
    print("      -> Creating Clean Phenotype Map...")
    
    # Safety: Ensure radius column exists
    if 'Refined Radius (um)' not in df.columns and 'radius' in df.columns:
        df['Refined Radius (um)'] = df['radius']
        
    # Filter Data
    plot_df = df[df['t_cortex'] > 0.05].copy()
    plot_df = plot_df[plot_df['Phenotype_Category'] != 'Excluded'] 
    
    if plot_df.empty: return

    plt.figure(figsize=(10, 8))
    
    # MAIN PLOT: Use get_phenotype_palette() for MFA Colors
    sns.scatterplot(
        data=plot_df, 
        x='A localization', 
        y='uniformity', 
        hue='Phenotype_Category',       # Color by Category
        style='Phenotype_Category',     # Shape by Category (Accessibility)
        size='Refined Radius (um)', 
        sizes=(40, 400), 
        palette=plotting.get_phenotype_palette(), 
        alpha=0.85, 
        edgecolor='white'
    )

    # BOUNDARIES: Use MFA Light Grey
    mfa_grey = plotting.MFA_COLORS['light_grey']
    mfa_dark = plotting.MFA_COLORS['dark_blue']
    mfa_med  = plotting.MFA_COLORS['medium_blue']

    plt.axvline(0.65, color=mfa_grey, linestyle='--', lw=2)
    plt.axhline(0.4, color=mfa_grey, linestyle='--', lw=2)
    
    # TEXT: Use MFA Blues for Semantic labels
    plt.text(0.66, 0.02, 'Uniform Zone', color=mfa_dark, fontweight='bold')
    plt.text(0.66, 1.4, 'Patchy (High Loc)', color=mfa_med)

    # Standard Formatting
    plt.title('Phenotype Map: Localization vs Uniformity')
    plt.xlabel('Localization (Higher is Better)')
    plt.ylabel('Uniformity (Lower is Better)')
    plt.xlim(left=-0.05, right=1.05)
    plt.ylim(bottom=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Phenotype")
    
    plotting.save_plot("Cortex_5D_Map.png", output_dir)

def plot_pairgrid(df, output_dir):
    print("      -> Creating Correlation Matrix...")
    
    # Safety: Ensure the radius column is named correctly
    if 'Refined Radius (um)' not in df.columns and 'radius' in df.columns:
        df['Refined Radius (um)'] = df['radius']
    
    # Define columns to include in the matrix
    cols = ['Refined Radius (um)', 't_cortex', 'rho_actin', 'A localization', 'uniformity', 'Phenotype_Category']
    
    plot_df = df[df['t_cortex'] > 0.05][cols].copy()
    plot_df = plot_df[plot_df['Phenotype_Category'].isin(plotting.get_phenotype_palette().keys())]
    
    if plot_df.empty: return

    # Apply Style manually since PairGrid creates its own figure
    plotting.set_paper_style(base_fontsize=12) 
    
    g = sns.pairplot(
        plot_df, hue='Phenotype_Category', 
        palette=plotting.get_phenotype_palette(),
        corner=True, plot_kws={'alpha': 0.7, 's': 40, 'edgecolor': 'white'}
    )
    g.fig.suptitle("Multi-Variable Correlation Matrix", y=1.02)
    g.savefig(os.path.join(output_dir, "Correlation_Matrix_PairPlot.png"), bbox_inches='tight')
    plt.close()


def run_phenotype_analysis(df, output_dir):
    print("\n--- Running Phenotype & Cortex Analysis (MFA Style) ---")
    
    # 1. Apply Global Style
    plotting.set_paper_style() 

    df = categorize_vesicles(df)
    
    # Save Log
    log_cols = ['Date', 'Name', 'Vesicle id', 'Phenotype_Category', 'A localization', 't_cortex', 'rho_actin', 'uniformity']
    existing_cols = [c for c in log_cols if c in df.columns]
    df[existing_cols].to_csv(os.path.join(output_dir, "Phenotype_Categorization_Log.csv"), index=False)
    
    plotting.plot_phenotype_composition(df, 'Category', 'Phenotype_Category', output_dir, 'Phenotype_Composition_Stacked.png')
    generate_phenotype_panel(df, output_dir)
    plot_pairgrid(df, output_dir)
    plot_5d_scatter(df, output_dir)
    save_phenotype_statistics(df, output_dir)