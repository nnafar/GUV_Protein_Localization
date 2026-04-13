# -*- coding: utf-8 -*-
"""
PLOTTING MODULE
Centralized visualization logic for all analysis modules.

CHANGES FROM PREVIOUS VERSION:
-------------------------------
1. NEW: plot_violin_scatter_box()
   Replaces the separate histogram + violin functions for size analysis.
   Layers three plots on top of each other:
     - Violin  : shows the full distribution shape
     - Box     : narrow box showing median + 25th/75th percentile lines
     - Strip   : individual data points (semi-transparent)

2. UPDATED: PHENO_PALETTE
   Added 'SHELL' color for the Factin condition.

3. UPDATED: generate_category_panel()
   Now handles SHELL/LUMENAL phenotypes for Factin, and skips
   phenotype-specific columns (like Gini_Index) gracefully if they
   are all NaN for a given condition.

4. UPDATED: plot_cortex_map() and plot_pairplot()
   Now handle Factin phenotypes (SHELL, LUMENAL) in addition to
   BranchedCortex/LinearCortex phenotypes.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents plots from popping up
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# =============================================================================
# 1. STYLE & COLORS
# =============================================================================

BlueScheme_COLORS = {
    'white_blue':     '#EDF2FB',
    'lightest_blue':  '#90A3B0',
    'Lighter_blue':   '#7A91A0',
    'light_blue':     '#647E90',
    'medium_blue':    '#4D6C81',
    'darker_blue':    '#375971',
    'darkest_blue':   '#214761',
}

# Color assigned to each phenotype label in all plots.
# Adding a new phenotype only requires adding one line here.
PHENO_PALETTE = {
    'EXCLUDED':   BlueScheme_COLORS['white_blue'],
    'EMPTY':      BlueScheme_COLORS['lightest_blue'],
    'LUMENAL':    BlueScheme_COLORS['light_blue'],   
    'SPARSE':     BlueScheme_COLORS['medium_blue'],
    'PATCHY':     BlueScheme_COLORS['darker_blue'],
    'CONTINUOUS': BlueScheme_COLORS['darkest_blue'],
    'SHELL':      BlueScheme_COLORS['Lighter_blue'],    
}

# Color assigned to each condition in size plots.
CONDITION_PALETTE = {
    'Empty':          BlueScheme_COLORS['lightest_blue'],
    'Factin':         BlueScheme_COLORS['light_blue'], 
    'BranchedCortex': BlueScheme_COLORS['medium_blue'],
    'LinearCortex':   BlueScheme_COLORS['darker_blue'],
}

def set_paper_style():
    """Sets a clean, publication-quality plot style."""
    sns.set_style("ticks")
    sns.set_context("talk", font_scale=1.0)
    plt.rcParams.update({'figure.dpi': 300})


# =============================================================================
# 2. HELPER: STATISTICAL ANNOTATION
# =============================================================================

def add_significance_bars(ax, data, x, y, order, pairs):
    """
    Adds significance bars (*/**/***/ns) above box plots.
    Only drawn when the Mann-Whitney U test gives p < 0.05.
    """
    y_vals = data[y].dropna()
    if y_vals.empty:
        return

    y_max   = y_vals.max()
    y_range = y_vals.max() - y_vals.min()
    if y_range == 0:
        return

    offset_step = y_range * 0.10
    current_y   = y_max + (y_range * 0.05)

    for p1, p2 in pairs:
        group1 = data[data[x] == p1][y].dropna()
        group2 = data[data[x] == p2][y].dropna()

        if len(group1) < 3 or len(group2) < 3:
            continue

        try:
            _, p = mannwhitneyu(group1, group2, alternative='two-sided')
        except Exception:
            continue

        if p >= 0.05:
            continue  # not significant — skip

        sig_symbol = '***' if p < 0.001 else '**' if p < 0.01 else '*'

        # Draw the horizontal bar + tick marks
        x1, x2   = order.index(p1), order.index(p2)
        bar_tips  = current_y - (y_range * 0.02)
        ax.plot([x1, x1, x2, x2],
                [bar_tips, current_y, current_y, bar_tips],
                lw=1.2, c='k')
        ax.text((x1 + x2) * 0.5, current_y, sig_symbol,
                ha='center', va='bottom', fontsize=10, color='k')
        current_y += offset_step


# =============================================================================
# 3. SIZE ANALYSIS PLOTS
# =============================================================================

def plot_violin_scatter_box(data, x_col, y_col, title, ylabel,
                             output_dir, filename):
    """
    Creates a combined violin + scatter + box plot — the most informative
    way to visualize a distribution alongside individual data points.

    How it works (think of it as three layers drawn on top of each other):
    -----------------------------------------------------------------------
    Layer 1 — Violin:
      The 'violin' shape is wide where many vesicles have that radius value,
      and narrow where few do. Like a histogram rotated 90° and mirrored.

    Layer 2 — Box:
      A narrow box drawn on top of the violin.
        • The middle line  = median (50th percentile — the "middle" value)
        • Box edges        = 25th and 75th percentile (middle 50% of data)
        • Whiskers         = extend to 1.5× the interquartile range
        (Outliers beyond the whiskers are hidden to keep the plot clean)

    Layer 3 — Strip:
      Every individual vesicle is shown as a small dot.
      They're randomly "jittered" (nudged left/right) so dots don't overlap.

    Parameters
    ----------
    data       : pandas DataFrame
    x_col      : column name for the x-axis (e.g. 'Category')
    y_col      : column name for the y-axis (e.g. 'Refined Radius (um)')
    title      : plot title string
    ylabel     : y-axis label string
    output_dir : folder to save the plot
    filename   : output filename (include .png)
    """
    print(f"  -> Generating violin-scatter-box plot: {filename}")

    # Determine the order and colors for the x-axis categories
    # We only include categories that are actually present in the data.
    desired_order = ['BranchedCortex', 'LinearCortex', 'Factin', 'Empty']
    order = [c for c in desired_order if c in data[x_col].unique()]
    palette = [CONDITION_PALETTE.get(c, BlueScheme_COLORS['darkest_blue']) for c in order]

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Layer 1: Violin ---
    # inner=None means don't draw anything inside the violin by default —
    # we'll add our own box on top for more control.
    sns.violinplot(
        data=data,
        x=x_col,
        y=y_col,
        order=order,
        palette=palette,
        inner=None,      # don't draw inner markers (we add box manually)
        alpha=0.35,      # semi-transparent so the box and dots are visible
        ax=ax,
        linewidth=1.5,
    )

    # --- Layer 2: Box ---
    # width=0.12 makes it very narrow so it sits cleanly inside the violin.
    # showfliers=False hides individual outlier dots from the box plot
    # (they'll be shown by the strip plot instead).
    sns.boxplot(
        data=data,
        x=x_col,
        y=y_col,
        order=order,
        palette=palette,
        width=0.12,
        showfliers=False,
        ax=ax,
        # Style the box and whisker lines
        boxprops=dict(alpha=0.85, linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(color='white', linewidth=2.5),  # white median line
    )

    # --- Layer 3: Strip (individual data points) ---
    # jitter=True nudges points left/right randomly to avoid overlap.
    # size=3 and alpha=0.45 keep them visible but not overwhelming.
    sns.stripplot(
        data=data,
        x=x_col,
        y=y_col,
        order=order,
        color='#222222',  # near-black dots
        size=3,
        alpha=0.45,
        jitter=True,
        ax=ax,
    )

    # ---- Labelling ----
    ax.set_title(title, pad=14)
    ax.set_xlabel('')       # x labels come from the category names
    ax.set_ylabel(ylabel)

    # Add N count below each condition label
    for i, cat in enumerate(order):
        n = data[data[x_col] == cat][y_col].dropna().shape[0]
        ax.text(i, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06,
                f"N={n}", ha='center', va='top', fontsize=10, color='#444444')

    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()
    print(f"     Saved: {filename}")


# =============================================================================
# 4. PHENOTYPE ANALYSIS PLOTS
# =============================================================================

def plot_phenotype_composition(df, x_col, hue_col, output_dir, filename):
    """
    Generates a 100% stacked bar chart showing phenotype composition
    for each condition (Category).

    Each bar totals 100%, and the segments show what fraction of vesicles
    in that condition belong to each phenotype.
    """
    print(f"  -> Generating Phenotype Composition chart: {filename}")

    counts = df.groupby([x_col, hue_col]).size().unstack(fill_value=0)
    # Convert to percentages (each row sums to 100%)
    props = counts.div(counts.sum(axis=1), axis=0) * 100

    # Use a consistent left-to-right phenotype order in the stacked bar
    desired_order = ['EMPTY', 'LUMENAL', 'SPARSE', 'PATCHY', 'CONTINUOUS',
                     'SHELL', 'EXCLUDED']
    existing_cols = [c for c in desired_order if c in props.columns]
    props = props[existing_cols]

    fig, ax = plt.subplots(figsize=(9, 6))
    props.plot(
        kind='bar',
        stacked=True,
        color=[PHENO_PALETTE.get(c, '#888888') for c in existing_cols],
        ax=ax,
        width=0.6,
        edgecolor='white',
        linewidth=0.5,
    )

    ax.set_title('Phenotype Composition by Condition')
    ax.set_ylabel('Percentage of Vesicles (%)')
    ax.set_xlabel('')
    ax.legend(title='Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=15, ha='right')
    sns.despine()

    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()


def generate_category_panel(df, output_dir):
    """
    Generates a grid of box + strip plots with significance bars.
    Rows = metrics, Columns = conditions.

    Only shows rows for metrics where data actually exists for that condition
    (so Gini_Index won't be empty for Factin just because it's in the list).
    """
    print("  -> Creating Phenotype Comparison Matrix...")

    # Radial_Kurtosis removed: insufficient data points (~17-21) in the
    # cortical window make kurtosis estimates unreliable, and it was
    # strongly collinear with ISM and Gini_Index anyway.
    metrics = [
        ('t_cortex',       'Cortex Thickness (µm)'),
        ('A localization', 'Localization (AU)'),
        ('ISM',            'ISM Index'),
        ('Gini_Index',     'Gini Coefficient'),
    ]

    # Phenotype order per condition.
    # LinearCortex has no PATCHY — Gini does not discriminate linear networks.
    phenotype_order_map = {
        'BranchedCortex': ['SPARSE', 'PATCHY', 'CONTINUOUS'],
        'LinearCortex':   ['SPARSE', 'CONTINUOUS'],
        'Factin':         ['SHELL', 'LUMENAL'],
    }

    stats_pairs_map = {
        'BranchedCortex': [('SPARSE', 'PATCHY'), ('PATCHY', 'CONTINUOUS'), ('SPARSE', 'CONTINUOUS')],
        'LinearCortex':   [('SPARSE', 'CONTINUOUS')],
        'Factin':         [('SHELL', 'LUMENAL')],
    }

    # Only show conditions that are in the dataframe
    categories = [c for c in ['BranchedCortex', 'LinearCortex', 'Factin']
                  if c in df['Category'].unique()]

    n_metrics = len(metrics)
    n_cats    = len(categories)

    if n_cats == 0:
        print("  ! No eligible conditions found for category panel.")
        return

    fig, axes = plt.subplots(
        nrows=n_metrics, ncols=n_cats,
        figsize=(5 * n_cats, 4 * n_metrics),
        sharey='row', sharex=False   # don't share x — each column has its own phenotype labels
    )

    # Normalise axes to always be a 2D array for consistent indexing
    if n_metrics == 1: axes = np.array([axes])
    if n_cats   == 1: axes = axes.reshape(-1, 1)

    for i, (metric_col, metric_label) in enumerate(metrics):
        for j, cat in enumerate(categories):
            ax     = axes[i, j]
            p_order  = phenotype_order_map.get(cat, [])
            s_pairs  = stats_pairs_map.get(cat, [])

            subset = df[
                (df['Category'] == cat) &
                (df['Phenotype_Category'].isin(p_order))
            ].copy()

            if not subset.empty and metric_col in subset.columns:
                sns.boxplot(
                    data=subset, x='Phenotype_Category', y=metric_col, ax=ax,
                    hue='Phenotype_Category', legend=False,
                    order=p_order, palette=PHENO_PALETTE,
                    width=0.5, showfliers=False,
                )
                sns.stripplot(
                    data=subset, x='Phenotype_Category', y=metric_col, ax=ax,
                    order=p_order, color='black', size=3, alpha=0.4,
                    jitter=True, dodge=False,
                )
                add_significance_bars(ax, subset, 'Phenotype_Category',
                                      metric_col, p_order, s_pairs)

            # Row and column labels
            if j == 0: ax.set_ylabel(metric_label, fontweight='bold')
            else:      ax.set_ylabel('')
            if i == 0: ax.set_title(cat, fontweight='bold', pad=15)

            ax.set_xlabel('')
            sns.despine(ax=ax)
            ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "Phenotype_Characteristics_Matrix_Stats.png"),
        bbox_inches='tight'
    )
    plt.close()


def plot_cortex_map(df, category_name, output_dir):
    """
    Scatter plot: localization (x) vs Gini / distribution spread (y),
    coloured by phenotype, sized by t_cortex.

    Works for both BranchedCortex/LinearCortex (SPARSE/PATCHY/CONTINUOUS)
    and Factin (SHELL/LUMENAL).
    """
    print(f"  -> Generating Cortex Map for: {category_name}")

    # Include whichever phenotypes are present for this condition
    valid_phenotypes = ['SPARSE', 'PATCHY', 'CONTINUOUS', 'SHELL', 'LUMENAL']
    subset = df[df['Phenotype_Category'].isin(valid_phenotypes)].dropna(
        subset=['A localization', 'Gini_Index'])

    if subset.empty:
        print(f"      ! No plottable data for {category_name} cortex map.")
        return

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=subset,
        x='A localization',
        y='Gini_Index',
        hue='Phenotype_Category',
        palette=PHENO_PALETTE,
        size='t_cortex',
        sizes=(20, 200),
        alpha=0.7,
        edgecolor='k',
        linewidth=0.4,
    )
    plt.title(f"Cortex Map: {category_name}")
    plt.xlabel('Localization Score')
    plt.ylabel('Gini Index (spatial heterogeneity)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()

    plt.savefig(
        os.path.join(output_dir, f"Cortex_Map_{category_name}.png"),
        bbox_inches='tight'
    )
    plt.close()


def plot_correlation_heatmap(corr_matrix, output_dir, filename_suffix=""):
    """Plots a Spearman correlation heatmap (lower triangle only)."""
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix, mask=mask,
        center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        cmap='vlag',
        square=True, linewidths=1,
        cbar_kws={"shrink": .5},
    )
    plt.title(f'Spearman Correlation Matrix{filename_suffix}')

    plt.savefig(
        os.path.join(output_dir, f"Spearman_Correlation_Heatmap{filename_suffix}.png"),
        bbox_inches='tight'
    )
    plt.close()


def plot_pairplot(df, output_dir, filename_suffix=""):
    """
    Generates a pair plot for key metrics, coloured by phenotype.
    Covers all phenotypes (SPARSE/PATCHY/CONTINUOUS for cortex conditions,
    SHELL/LUMENAL for Factin).
    """
    print(f"  -> Generating Pair Plot: {filename_suffix}")

    # Radial_Kurtosis removed (unreliable — see generate_category_panel).
    # PATCHY removed from valid_phenotypes for LinearCortex: since the
    # classifier no longer assigns PATCHY to LinearCortex rows, it will
    # never appear in that condition's data. Keeping it here for
    # BranchedCortex where it remains a valid phenotype.
    metrics = ['t_cortex', 'A localization', 'ISM', 'Gini_Index']

    valid_phenotypes = ['SPARSE', 'PATCHY', 'CONTINUOUS', 'SHELL', 'LUMENAL']
    subset = df[df['Phenotype_Category'].isin(valid_phenotypes)].dropna(subset=metrics)

    if subset.empty or len(subset) < 5:
        print("      ! Not enough data for pair plot.")
        return

    g = sns.pairplot(
        subset,
        vars=metrics,
        hue='Phenotype_Category',
        palette=PHENO_PALETTE,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30},
        diag_kws={'fill': True},
    )
    g.fig.suptitle(f"Metric Pair Plot {filename_suffix}", y=1.02)
    g.savefig(
        os.path.join(output_dir, f"Correlation_Matrix_PairPlot{filename_suffix}.png"),
        bbox_inches='tight'
    )
    plt.close()