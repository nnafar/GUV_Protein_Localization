# -*- coding: utf-8 -*-
"""
PLOTTING MODULE
Centralized plotting configuration using the Blue-to-Red color scheme.
"""

import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from typing import Tuple, List


# =============================================================================
# 1. BlUE-TO-RED STYLE DEFINITIONS
# =============================================================================

MFA_COLORS = {
    # Reference Palette (Hex)
    'dark_blue':   '#1065AB', # R 016, G 101, B 171
    'medium_blue': '#3A93C3', # R 058, G 147, B 195
    'light_blue':  '#8EC4DE', # R 142, G 196, B 222
    'pale_blue':   '#D1E5F0', # R 209, G 229, B 240
    'light_grey':  '#E7E6E3', # R 231, G 230, B 227
    'white':       '#F9F9F9', # R 249, G 249, B 249
    'pale_red':    '#FDDBC7', # R 254, G 219, B 199
    'light_red':   '#F6A482', # R 246, G 164, B 130
    'medium_red':  '#D75F4C', # R 215, G 095, B 076
    'dark_red':    '#B31529', # R 179, G 021, B 041
    
    # Semantic Mapping for Plots
    'grid':        '#D1E5F0', 
    'primary':     '#000000',
}

# Define the Phenotype Mapping (Ensures consistency across all plots)
PHENOTYPE_PALETTE = {
    'Uniform':       MFA_COLORS['dark_blue'],
    'Patchy':        MFA_COLORS['medium_blue'],
    'Sparse':         MFA_COLORS['light_red'],
    'Lumenal Actin': MFA_COLORS['dark_red'],
    'Empty':         MFA_COLORS['light_grey'],
    'Excluded':      '#555555' # Dark Grey
}

def set_paper_style(base_fontsize: int = 14, dpi: int = 300) -> None:
    """Applies publication-quality style to Matplotlib plots."""
    plt.rcdefaults() 
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': base_fontsize,
        'axes.labelsize': base_fontsize,
        'axes.titlesize': base_fontsize + 2,
        'axes.titleweight': 'bold',
        'legend.fontsize': base_fontsize - 2,
        'legend.frameon': False, # Cleaner look
        'figure.dpi': dpi,
        'figure.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.5,
        'grid.color': MFA_COLORS['grid'],
        'grid.linestyle': '--',
        'lines.linewidth': 2.0,
    })

def get_phenotype_palette():
    """Returns the dictionary mapping phenotypes to MFA colors."""
    return PHENOTYPE_PALETTE

def get_gradient_cmap():
    """Returns a LinearSegmentedColormap from Blue to Red."""
    colors = [MFA_COLORS['dark_blue'], MFA_COLORS['light_blue'], 
              MFA_COLORS['pale_red'], MFA_COLORS['dark_red']]
    return mpl.colors.LinearSegmentedColormap.from_list("mfa_gradient", colors)

# =============================================================================
# 2. PLOTTING FUNCTIONS
# =============================================================================

def save_plot(filename, output_dir):
    """Helper to save and close plots."""
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


def plot_histogram(data, column, title, xlabel, output_dir, filename, color=None):
    """Generates a standard Histogram with KDE."""
    if color is None: color = MFA_COLORS['medium_blue']
    
    plt.figure(figsize=(8, 6))
    mean_val = np.mean(data[column])
    
    sns.histplot(data=data, x=column, kde=True, color=color, edgecolor='white', alpha=0.8)
    plt.axvline(mean_val, color=MFA_COLORS['dark_red'], linestyle='--', lw=2, label=f'Mean: {mean_val:.2f}')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y')
    
    save_plot(filename, output_dir)


def plot_violin_comparison(data, x_col, y_col, title, ylabel, output_dir, filename):
    """Generates a Violin plot to show density distribution."""
    plt.figure(figsize=(10, 6))
    
    # Use MFA colors if not Phenotype
    if x_col == 'Phenotype_Category':
        palette = PHENOTYPE_PALETTE
    else:
        # Default to MFA Blues for experimental categories
        palette = [MFA_COLORS['medium_blue'], MFA_COLORS['light_blue'], 
                   MFA_COLORS['dark_blue'], MFA_COLORS['pale_blue']]
    
    sns.violinplot(data=data, x=x_col, y=y_col, hue=x_col, legend=False, 
                   palette=palette, inner="quartile", alpha=0.8, linewidth=1)
    
    sns.stripplot(data=data, x=x_col, y=y_col, color='black', size=3, alpha=0.2, jitter=True)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    
    save_plot(filename, output_dir)

def get_experiment_palette():
    """Returns the standard blues for non-phenotype comparisons."""
    return [MFA_COLORS['medium_blue'], MFA_COLORS['light_blue'], 
            MFA_COLORS['dark_blue'], MFA_COLORS['pale_blue']]
    
def plot_boxplot_comparison(data, x_col, y_col, title, ylabel, output_dir=None, filename=None, ax=None, order=None):
    """Generates a boxplot comparing a numerical variable across categories."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        is_standalone = True
    else:
        is_standalone = False

    # Use the centralized palette
    if x_col == 'Phenotype_Category':
        palette = PHENOTYPE_PALETTE
    else:
        # Match the Violin Plot logic (Cycle through MFA Blues)
        palette = get_experiment_palette()

    # Pass the 'order' parameter to seaborn
    sns.boxplot(data=data, x=x_col, y=y_col, hue=x_col, legend=False, ax=ax, 
                palette=palette, order=order, boxprops=dict(alpha=0.8), showfliers=False)
    
    sns.stripplot(data=data, x=x_col, y=y_col, color='black', alpha=0.3, jitter=True, size=3, ax=ax, order=order)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')

    if is_standalone and filename and output_dir:
         save_plot(filename, output_dir)


def plot_phenotype_composition(data, category_col, phenotype_col, output_dir, filename):
    """Generates a 100% Stacked Bar Plot."""
    counts = data.groupby([category_col, phenotype_col]).size().unstack(fill_value=0)
    percents = counts.div(counts.sum(axis=1), axis=0) * 100
    
    # Map colors manually to match the stack
    stack_colors = [PHENOTYPE_PALETTE.get(col, '#999999') for col in percents.columns]

    plt.figure(figsize=(10, 6))
    ax = percents.plot(kind='bar', stacked=True, color=stack_colors, 
                       alpha=0.9, edgecolor='white', rot=45, width=0.7)
    
    plt.title('Phenotypic Composition by Category')
    plt.xlabel('Experimental Category')
    plt.ylabel('Percentage of Vesicles (%)')
    plt.legend(title='Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y')
    
    fig = plt.gcf()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Plot saved: {filename}")

def plot_correlation_heatmap(corr_matrix, output_dir):
    """Plots a Spearman Correlation Heatmap using MFA Diverging Colors."""
    plt.figure(figsize=(10, 8))
    
    # Custom Diverging Palette (Blue -> White -> Red)
    colors = [MFA_COLORS['dark_blue'], MFA_COLORS['white'], MFA_COLORS['dark_red']]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mfa_diverging", colors)
    
    # Mask the upper triangle (redundant info)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap, 
        center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", 
        square=True, linewidths=1, linecolor='white',
        cbar_kws={"shrink": .7, "label": "Spearman Correlation (r)"}
    )
    
    plt.title('Spearman Correlation Matrix (Non-Parametric)', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_plot("Spearman_Correlation_Heatmap.png", output_dir)