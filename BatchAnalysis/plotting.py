# -*- coding: utf-8 -*-
"""
PLOTTING MODULE
Centralized plotting configuration. 
Contains ALL visualization logic.
"""

import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# =============================================================================
# 1. STYLE & COLORS
# =============================================================================

MFA_COLORS = {
    'dark_blue':   '#1065AB',
    'medium_blue': '#3A93C3',
    'light_blue':  '#8EC4DE',
    'light_grey':  '#E7E6E3',
    'medium_grey': '#B0B0B0',
    'dark_grey':   '#606060',
    'white':       '#FFFFFF',
    'dark_red':    '#B2182B',
    'pale_red':    '#FDDBC7'
}

PHENO_PALETTE = {
    'SPARSE': MFA_COLORS['light_blue'],
    'PATCHY': MFA_COLORS['medium_blue'],
    'CONTINUOUS': MFA_COLORS['dark_blue'],
    'EMPTY': MFA_COLORS['light_grey'],
    'LUMENAL': MFA_COLORS['medium_grey'],
    'EXCLUDED': '#000000'
}

def set_paper_style():
    """Sets seaborn style for publication-quality plots."""
    sns.set_style("ticks")
    sns.set_context("talk", font_scale=1.0)
    plt.rcParams.update({'figure.dpi': 300})

# =============================================================================
# 2. HELPER: STATISTICAL ANNOTATION
# =============================================================================

def add_significance_bars(ax, data, x, y, order, pairs):
    """
    Runs Mann-Whitney U test and adds significance bars to the plot.
    """
    y_max = data[y].max()
    if pd.isna(y_max): return
    
    y_range = data[y].max() - data[y].min()
    offset_step = y_range * 0.1
    current_y = y_max + (y_range * 0.05)

    for p1, p2 in pairs:
        group1 = data[data[x] == p1][y].dropna()
        group2 = data[data[x] == p2][y].dropna()

        if len(group1) > 2 and len(group2) > 2:
            try:
                stat, p = mannwhitneyu(group1, group2)
                if p < 0.05:
                    sig_symbol = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                    
                    x1, x2 = order.index(p1), order.index(p2)
                    bar_h = current_y
                    bar_tips = bar_h - (y_range * 0.02)
                    
                    ax.plot([x1, x1, x2, x2], [bar_tips, bar_h, bar_h, bar_tips], lw=1.2, c='k')
                    ax.text((x1 + x2) * 0.5, bar_h, sig_symbol, ha='center', va='bottom', fontsize=10, color='k')
                    
                    current_y += offset_step
            except Exception:
                pass

# =============================================================================
# 3. SIZE ANALYSIS PLOTS
# =============================================================================

def plot_histogram(data, column, title, xlabel, output_dir, filename, hue=None):
    """Plots a histogram comparing categories."""
    print(f"      -> Generating Histogram: {filename}")
    plt.figure(figsize=(8, 6))
    
    sns.histplot(
        data=data, x=column, hue=hue, 
        element="step", fill=True, stat="density", common_norm=False,
        palette=[MFA_COLORS['medium_blue'], MFA_COLORS['dark_grey']]
    )
    
    plt.title(title)
    plt.xlabel(xlabel)
    sns.despine()
    
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()

def plot_violin_comparison(data, x_col, y_col, title, ylabel, output_dir, filename):
    """Plots a violin plot comparing size distributions."""
    print(f"      -> Generating Violin Plot: {filename}")
    plt.figure(figsize=(6, 6))
    
    cats = data[x_col].unique()
    pal = [MFA_COLORS['medium_blue'], MFA_COLORS['light_blue']] if len(cats) <= 2 else "Blues"

    sns.violinplot(
        data=data, x=x_col, y=y_col, hue=x_col, legend=False,
        palette=pal, inner="quartile", alpha=0.5
    )
    sns.stripplot(
        data=data, x=x_col, y=y_col, color='k', alpha=0.3, size=2, jitter=True
    )
    
    plt.title(title)
    plt.ylabel(ylabel)
    sns.despine()
    
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()

# =============================================================================
# 4. PHENOTYPE ANALYSIS PLOTS
# =============================================================================

def plot_phenotype_composition(df, x_col, hue_col, output_dir, filename):
    """Generates a 100% stacked bar chart of phenotype composition."""
    print(f"      -> Generating Composition Plot: {filename}")
    
    counts = df.groupby([x_col, hue_col]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0) * 100
    
    desired_order = ['EMPTY', 'LUMENAL', 'SPARSE', 'PATCHY', 'CONTINUOUS']
    existing_cols = [c for c in desired_order if c in props.columns]
    props = props[existing_cols]

    ax = props.plot(kind='bar', stacked=True, color=[PHENO_PALETTE.get(x, '#333') for x in existing_cols], 
                    figsize=(8, 6), width=0.8)
    
    plt.title('Phenotype Composition by Category')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    sns.despine()
    
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()


def generate_category_panel(df, output_dir):
    """
    Generates a Matrix of Box Plots with Stats (Rows=Metrics, Cols=Categories).
    """
    print("      -> Creating Phenotype Comparison Matrix...")
    
    # ADDED: Radial Kurtosis to the list of metrics
    metrics = [
        ('t_cortex', 'Cortex Thickness (µm)'),
        ('A localization', 'Localization (AU)'),
        ('ISM', 'ISM Index'),
        ('Gini_Index', 'Gini Coefficient'),
        ('Radial_Kurtosis', 'Radial Kurtosis')
    ]
    
    categories = sorted(df['Category'].unique())
    phenotype_order = ['SPARSE', 'PATCHY', 'CONTINUOUS']
    stats_pairs = [('SPARSE', 'PATCHY'), ('PATCHY', 'CONTINUOUS'), ('SPARSE', 'CONTINUOUS')]
    
    n_metrics, n_cats = len(metrics), len(categories)
    
    # Adjusted figsize to accommodate the 5th row
    fig, axes = plt.subplots(nrows=n_metrics, ncols=n_cats, 
                             figsize=(5 * n_cats, 4 * n_metrics), 
                             sharey='row', sharex=True)
    
    if n_metrics == 1: axes = np.array([axes])
    if n_cats == 1: axes = axes.reshape(-1, 1)
    if n_metrics == 1 and n_cats == 1: axes = np.array([[axes]])

    for i, (metric_col, metric_label) in enumerate(metrics):
        for j, cat in enumerate(categories):
            ax = axes[i, j]
            subset = df[(df['Category'] == cat) & (df['Phenotype_Category'].isin(phenotype_order))].copy()
            
            if not subset.empty:
                sns.boxplot(data=subset, x='Phenotype_Category', y=metric_col, ax=ax,
                            hue='Phenotype_Category', legend=False,
                            order=phenotype_order, palette=PHENO_PALETTE, width=0.5, showfliers=False)
                
                sns.stripplot(data=subset, x='Phenotype_Category', y=metric_col, ax=ax,
                              order=phenotype_order, color='black', size=3, alpha=0.4, jitter=True, dodge=False)
                
                add_significance_bars(ax, subset, 'Phenotype_Category', metric_col, phenotype_order, stats_pairs)

            if j == 0: ax.set_ylabel(metric_label, fontweight='bold')
            else: ax.set_ylabel("")
            if i == 0: ax.set_title(cat, fontweight='bold', pad=15)
            
            ax.set_xlabel("")
            sns.despine(ax=ax)
            ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Phenotype_Characteristics_Matrix_Stats.png"), bbox_inches='tight')
    plt.close()


def plot_cortex_map(df, category_name, output_dir):
    """Plots the 5D scatter map."""
    print(f"      -> Generating Map for: {category_name}")
    plt.figure(figsize=(7, 6))
    
    subset = df[df['Phenotype_Category'].isin(['SPARSE', 'PATCHY', 'CONTINUOUS'])]
    if subset.empty: return

    sns.scatterplot(
        data=subset, x='A localization', y='Gini_Index',
        hue='Phenotype_Category', palette=PHENO_PALETTE,
        size='t_cortex', sizes=(20, 200), alpha=0.7, edgecolor='k'
    )
    
    plt.title(f"Cortex Map: {category_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    plt.savefig(os.path.join(output_dir, f"Cortex_Map_{category_name}.png"), bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(corr_matrix, output_dir, filename_suffix=""):
    """Plots Spearman Correlation Heatmap."""
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, mask=mask, center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", cmap='vlag',
        square=True, linewidths=1, cbar_kws={"shrink": .5}
    )
    plt.title(f'Spearman Correlation Matrix {filename_suffix}')
    plt.savefig(os.path.join(output_dir, f"Spearman_Correlation_Heatmap{filename_suffix}.png"), bbox_inches='tight')
    plt.close()

def plot_pairplot(df, output_dir, filename_suffix=""):
    """Generates a Pair Plot for key metrics, colored by Phenotype."""
    print(f"      -> Generating Pair Plot: {filename_suffix}")
    
    # ADDED: Radial Kurtosis
    metrics = ['t_cortex', 'A localization', 'ISM', 'Gini_Index', 'Radial_Kurtosis']
    
    subset = df[df['Phenotype_Category'].isin(['SPARSE', 'PATCHY', 'CONTINUOUS'])].dropna(subset=metrics)
    
    if subset.empty:
        print("      ! Not enough data for pairplot.")
        return

    g = sns.pairplot(
        subset, 
        vars=metrics, 
        hue='Phenotype_Category',
        palette=PHENO_PALETTE,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30},
        diag_kws={'fill': True}
    )
    
    g.fig.suptitle(f"Metric Pair Plot {filename_suffix}", y=1.02)
    g.savefig(os.path.join(output_dir, f"Correlation_Matrix_PairPlot{filename_suffix}.png"), bbox_inches='tight')
    plt.close()