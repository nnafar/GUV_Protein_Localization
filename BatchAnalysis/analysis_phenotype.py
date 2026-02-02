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
    """Classifies vesicles using a Size-Dependent Thickness criterion with a Buffer Zone."""
    
    # --- CONSTANTS ---
    NO_LOC_LIMIT = 0.4
    SPARSE_LOC_LIMIT = 0.5
    PATCHY_LOC_LIMIT = 0.65
    
    RHO_LIMIT = 2
    UNIFORMITY_LIMIT = 0.4
    SPARSE_MAX_THICK = 1.6  # RESTORED: General limit for sparse signal
    
    # --- SIZE-DEPENDENT LOGIC CONSTANTS ---
    BUFFER_MIN = 2.2         # Lower bound of intermediate size zone
    BUFFER_MAX = 2.8         # Upper bound of intermediate size zone
    STRICT_THICK_LIMIT = 1.2 # Limit for SMALL vesicles (< 2.2 um)
    RELAXED_THICK_LIMIT = 1.6 # Limit for LARGE vesicles (> 2.8 um)

    def classify(row):
        cat = str(row['Category'])
        loc = row['A localization']
        rho = row['rho_actin']
        thick = row['t_cortex']
        cv = row['uniformity']
        rad = row['Refined Radius (um)']
        
        # 1. Base Exclusion
        if pd.isna(loc) or pd.isna(rho) or pd.isna(thick):
            return "Excluded"
            
        # 2. Low Localization (< 0.4)
        if loc < NO_LOC_LIMIT:
            return "Empty" if rho < RHO_LIMIT else "Lumenal Actin"
            
        # 3. Sparse Branch (0.4 - 0.5)
        if loc < SPARSE_LOC_LIMIT:
            # RESTORED: Original Sparse Thickness Guardrail
            if thick > SPARSE_MAX_THICK:
                return "Excluded"
            
            if thick == 0.0:
                return "Lumenal Actin"
                
            return "Sparse"
            
        # 4. Patchy/Cortex Branch (0.5 - 0.65)
        if loc < PATCHY_LOC_LIMIT:
            if thick == 0.0:
                return "Lumenal Actin"
            
            # --- BRANCHED RULE (Size Dependent) ---
            if "Branched" in cat:
                # 4a. Identify ambiguous cases first
                if BUFFER_MIN <= rad <= BUFFER_MAX:
                    return "Intermediate"
                
                # 4b. Assign dynamic limits for non-ambiguous cases
                if rad < BUFFER_MIN:
                    max_allowed = STRICT_THICK_LIMIT
                else:
                    max_allowed = RELAXED_THICK_LIMIT
                
                # Check against the dynamic limit
                if thick > max_allowed:
                    return "Excluded"
                return "Patchy"
            
            # --- LINEAR RULE ---
            else:
                # RESTORED: Linear filaments can be thicker than branched ones
                return "Patchy"

        # 5. High Localization (> 0.65)
        else:
            if pd.isna(cv): return "Patchy"
            return "Uniform" if cv < UNIFORMITY_LIMIT else "Patchy"

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

def generate_category_panel(df, output_dir):
    """
    Comparison Panel: Experimental Category on X-Axis.
    Split by Phenotype (Hue).
    Includes STATISTICAL ANNOTATION and filters 'Empty' from Uniformity.
    """
    print("      -> Creating 2x2 Category Comparison Panel (Grouped + Stats)...")
    plot_df = df[df['Phenotype_Category'] != 'Excluded'].copy()
    
    # Define which pairs we want to test for significance
    # We focus on the most relevant cortical transition: Patchy vs Uniform
    stats_pairs = [
        ('Patchy', 'Uniform'),
        ('Sparse', 'Patchy')
    ]
    
    params = [
        ('A localization', 'A Localization'),
        ('t_cortex', 'Cortex Thickness'),
        ('rho_actin', 'Actin Density'),
        ('uniformity', 'Uniformity (CV)')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten() 

    for i, (col, title) in enumerate(params):
        ax = axes[i]
        
        if col in plot_df.columns:
            # --- INTELLIGENT DATA FILTERING ---
            # 1. Thickness: Remove Empty/Lumenal (No cortex to measure)
            if col == 't_cortex':
                 sub_df = plot_df[~plot_df['Phenotype_Category'].isin(['Empty', 'Lumenal Actin'])]
            
            # 2. Uniformity: Remove Empty (Noise) <--- THIS FIXES THE SCALE ISSUE
            elif col == 'uniformity':
                 sub_df = plot_df[~plot_df['Phenotype_Category'].isin(['Empty'])]
            
            # 3. Others: Keep all
            else:
                 sub_df = plot_df

            plotting.plot_boxplot_comparison(
                data=sub_df, 
                x_col='Category',   
                y_col=col, 
                hue='Phenotype_Category', 
                title=title, 
                ylabel=col, 
                ax=ax,
                box_pairs=stats_pairs # Pass the pairs to compare!
            )
            
            # Clean up
            ax.set_xlabel('') 
            if i < 2: ax.set_xlabel('') # Only show X labels on bottom row usually, but we rotate them anyway
        else:
            ax.text(0.5, 0.5, "Data Not Found", ha='center')

    plt.tight_layout()
    plotting.save_plot("Phenotype_Characteristics_Panel_2x2.png", output_dir)


def plot_5d_scatter(df, output_dir, filename_suffix=""):
    """
    Creates the 5D scatter map.
    If 'filename_suffix' is provided, it saves a separate file (e.g., per category).
    """
    if 'Refined Radius (um)' not in df.columns and 'radius' in df.columns:
        df['Refined Radius (um)'] = df['radius']
        
    plot_df = df[df['t_cortex'] > 0.05].copy()
    plot_df = plot_df[plot_df['Phenotype_Category'] != 'Excluded'] 
    
    if plot_df.empty: return

    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(
        data=plot_df, x='A localization', y='uniformity', 
        hue='Phenotype_Category', style='Phenotype_Category',
        size='Refined Radius (um)', sizes=(40, 400), 
        palette=plotting.get_phenotype_palette(), alpha=0.85, edgecolor='white'
    )

    # Boundaries
    mfa_grey = plotting.MFA_COLORS['light_grey']
    mfa_dark = plotting.MFA_COLORS['dark_blue']
    plt.axvline(0.65, color=mfa_grey, linestyle='--', lw=2)
    plt.axhline(0.4, color=mfa_grey, linestyle='--', lw=2)
    plt.text(0.66, 0.02, 'Uniform Zone', color=mfa_dark, fontweight='bold')

    plt.title(f'Phenotype Map: {filename_suffix.strip("_")}')
    plt.xlim(left=-0.05, right=1.05)
    plt.ylim(bottom=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Phenotype")
    
    plotting.save_plot(f"Cortex_5D_Map{filename_suffix}.png", output_dir)

def plot_pairgrid(df, output_dir, filename_suffix=""):
    """
    Creates correlation matrix (PairPlot).
    """
    if 'Refined Radius (um)' not in df.columns and 'radius' in df.columns:
        df['Refined Radius (um)'] = df['radius']
    
    cols = ['Refined Radius (um)', 't_cortex', 'rho_actin', 'A localization', 'uniformity', 'Phenotype_Category']
    plot_df = df[df['t_cortex'] > 0.05][cols].copy()
    plot_df = plot_df[plot_df['Phenotype_Category'].isin(plotting.get_phenotype_palette().keys())]
    
    if plot_df.empty or len(plot_df) < 5: return # Skip if too little data

    plotting.set_paper_style(base_fontsize=12) 
    
    g = sns.pairplot(
        plot_df, hue='Phenotype_Category', 
        palette=plotting.get_phenotype_palette(),
        corner=True, plot_kws={'alpha': 0.7, 's': 40, 'edgecolor': 'white'}
    )
    g.fig.suptitle(f"Multi-Variable Correlation {filename_suffix}", y=1.02)
    g.savefig(os.path.join(output_dir, f"Correlation_Matrix_PairPlot{filename_suffix}.png"), bbox_inches='tight')
    plt.close()


def run_phenotype_analysis(df, output_dir):
    print("\n--- Running Phenotype & Cortex Analysis (Category Split) ---")
    plotting.set_paper_style() 
    
    # 1. Run Classification Logic
    df = categorize_vesicles(df)
    
    # 1a. Save classification csv
    detailed_path = os.path.join(output_dir, "Vesicle_Phenotype_Assignments.csv")
    df.to_csv(detailed_path, index=False)
    print(f"      -> Detailed Assignments saved: {detailed_path}")
    
    # 1b. Composition Plot (Stacked Bar Chart)
    plotting.plot_phenotype_composition(
        df, 
        'Category', 
        'Phenotype_Category', 
        output_dir, 
        'Phenotype_Composition_Stacked.png'
    )
    
    # 2. Comparison Panel (X-Axis = Category)
    generate_category_panel(df, output_dir)
    
    # 3. SPLIT PLOTS: Loop through each Category and generate separate maps/matrices
    unique_cats = df['Category'].unique()
    print(f"      -> Generating separate maps for: {unique_cats}")
    
    for cat in unique_cats:
        cat_df = df[df['Category'] == cat].copy()
        suffix = f"_{cat}"
        
        # A. 5D Map per Category
        plot_5d_scatter(cat_df, output_dir, filename_suffix=suffix)
        
        # B. PairGrid per Category
        plot_pairgrid(cat_df, output_dir, filename_suffix=suffix)
        
    save_phenotype_statistics(df, output_dir)