# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import mannwhitneyu 
import plotting

# ----------------- PHYSICAL THRESHOLDS -----------------
NO_LOC_LIMIT = 0.40         # Below this is strictly No Localization
CORTEX_LOC_MIN = 0.60       # 
CORTEX_THICKNESS_MIN = 0.80 # Above this is the 'Dense/Thin' cortex
RHO_LIMIT = 2.0             # Baseline for completely empty signal
UNIFORMITY_MIN = 0.4
# -------------------------------------------------------

def categorize_vesicles(df):
    """Classifies vesicles based on the decision tree."""
    def classify(row):
        loc = row['A localization']
        rho = row['rho_actin']
        thick = row['t_cortex']
        cv = row['uniformity'] 
        
        if pd.isna(loc) or pd.isna(rho) or pd.isna(thick):
            return "Excluded"

        if loc < NO_LOC_LIMIT:
            if rho < RHO_LIMIT: return "Empty"
            else: return "Lumenal Actin"
            
        if thick >= CORTEX_THICKNESS_MIN:
            if loc < CORTEX_LOC_MIN: return "Fuzzy"
            else: return "Thick and uniform"
        else:
            if pd.isna(cv): return "Unclassified Thin"
            if cv >= UNIFORMITY_MIN: return "Thin and patchy"
            else: return "Thin and uniform"

    df['Phenotype_Category'] = df.apply(classify, axis=1)
    return df

def save_phenotype_statistics(df, output_dir):
    """Calculates summary stats for phenotypes."""
    print("     -> Calculating Phenotype Statistics...")
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
    level: 0 = standard low bracket. 1, 2, 3... = stacked higher brackets.
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
    
    # 4. Determine Height (Robust)
    # Find the indices spanned by the bracket (from min(x1,x2) to max(x1,x2))
    start, end = min(x1, x2), max(x1, x2)
    
    # Identify which categories are sitting 'under' this bracket
    categories_under_bracket = [order[i] for i in range(start, end+1)]
    
    # Find the maximum Y-value among ALL these categories to avoid cutting through data
    subset = df[df[x_col].isin(categories_under_bracket)][y_col]
    if subset.empty: y_max = 0
    else: y_max = subset.max()
    
    # Calculate spacing based on data range
    y_range = df[y_col].max() - df[y_col].min()
    if y_range == 0: y_range = 1
    
    # Stack height: Base + (Level * Step)
    base_clearance = y_range * 0.05
    step = y_range * 0.10
    
    y_h = y_max + base_clearance + (level * step)
    y_text = y_h + (y_range * 0.02)
    
    # 5. Draw
    ax.plot([x1, x1, x2, x2], [y_h - (y_range*0.02), y_h, y_h, y_h - (y_range*0.02)], lw=1.5, c='black')
    ax.text((x1+x2)*.5, y_text, sig, ha='center', va='bottom', color='black', fontsize=11, fontweight='bold')

def generate_phenotype_panel(df, output_dir):
    """Creates 2x2 Panel with STACKED statistical brackets."""
    print("     -> Creating 2x2 Phenotype Panel (with stacked stats)...")
    
    plot_df = df[df['Phenotype_Category'] != 'Excluded'].copy()
    
    order = ["Thin and patchy", "Thin and uniform", "Fuzzy", "Thick and uniform", "Lumenal Actin", "Empty"]
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
            
            # --- LEVEL 0: Local Comparisons ---
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, "Thin and patchy", "Thin and uniform", existing_order, level=0)
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, "Fuzzy", "Thick and uniform", existing_order, level=0)
            
            # --- LEVEL 1: Cross-Group Comparison (The Bridge) ---
            # Compares the two "Successful/Stable" states
            add_stat_annotation(ax, plot_df, 'Phenotype_Category', col, "Thin and uniform", "Thick and uniform", existing_order, level=1)

            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Phenotype', fontsize=11)
            ax.set_ylabel(col, fontsize=11)
            ax.tick_params(axis='x', rotation=45) 
            # Increase top margin significantly to fit the stacked brackets
            ax.set_ylim(top=ax.get_ylim()[1] * 1.25) 
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, f"Column '{col}' not found", ha='center')

    plt.tight_layout()
    plotting.save_plot("Phenotype_Characteristics_Panel_2x2.png", output_dir)

def plot_5d_scatter(df, output_dir):
    """Generates 5D Scatter Plot."""
    print("     -> Creating 5D Spatial Map...")
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
    print("     -> Creating Correlation Matrix...")
    cols = ['t_cortex', 'rho_actin', 'A localization', 'uniformity', 'Phenotype_Category']
    plot_df = df[df['t_cortex'] > 0.05][cols].copy()
    if plot_df.empty: return

    g = sns.pairplot(plot_df, hue='Phenotype_Category', palette='tab10', corner=True, plot_kws={'alpha': 0.6, 's': 40})
    g.fig.suptitle("Multi-Variable Correlation Matrix", y=1.02, fontsize=16)
    path = os.path.join(output_dir, "Correlation_Matrix_PairPlot.png")
    g.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: Correlation_Matrix_PairPlot.png")

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