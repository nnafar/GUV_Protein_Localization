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
from scipy.stats import mannwhitneyu
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

def add_grouped_significance_brackets(ax, df, x_col, y_col, hue_col, box_pairs, order, hue_order):
    """
    Draws statistical significance brackets on grouped boxplots.
    
    Parameters:
    ax: The matplotlib axis
    df: DataFrame
    x_col: The column for X-axis groups (Category)
    y_col: The value column
    hue_col: The column for sub-groups (Phenotype)
    box_pairs: List of tuples, e.g., [('Patchy', 'Uniform')] to compare within each X-group.
    order: List of X-axis categories
    hue_order: List of hue categories (must match plot)
    """
    
    # Constants for bracket placement
    y_max_global = df[y_col].max()
    y_range = df[y_col].max() - df[y_col].min()
    if y_range == 0: y_range = 1
    
    # Calculate bar positions
    n_hues = len(hue_order)
    width = 0.8  # Standard seaborn width
    slot_width = width / n_hues
    
    # Iterate over each X-axis Category (Branched, Linear, etc.)
    for x_idx, category in enumerate(order):
        
        # Track stacking height to avoid brackets overlapping
        level = 0 
        
        for (h1, h2) in box_pairs:
            # 1. Get Data
            try:
                # Find indices in the hue_order to calculate X position
                h1_idx = hue_order.index(h1)
                h2_idx = hue_order.index(h2)
            except ValueError:
                continue # Skip if phenotype not in this plot
            
            # Extract values for this specific Category + Phenotype combination
            data1 = df[(df[x_col] == category) & (df[hue_col] == h1)][y_col].dropna()
            data2 = df[(df[x_col] == category) & (df[hue_col] == h2)][y_col].dropna()
            
            # Need minimal data points
            if len(data1) < 3 or len(data2) < 3: 
                continue

            # 2. Run Statistics (Mann-Whitney U)
            stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
            
            if p >= 0.05: continue # Skip non-significant
            
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"

            # 3. Calculate Coordinates
            # Formula: center + (index - middle_offset) * slot_width
            x_center = x_idx
            x1 = x_center + (h1_idx - (n_hues - 1) / 2) * slot_width
            x2 = x_center + (h2_idx - (n_hues - 1) / 2) * slot_width
            
            # Find Y height (max of the two boxes + clearance)
            y_h = max(data1.max(), data2.max()) + (y_range * 0.05) + (level * y_range * 0.1)
            y_text = y_h + (y_range * 0.02)
            
            # 4. Draw Bracket
            # Legs: [x, x], [y_leg, y_top]
            col = 'black'
            ax.plot([x1, x1, x2, x2], [y_h, y_h + (y_range*0.02), y_h + (y_range*0.02), y_h], lw=1, c=col)
            
            # Star
            ax.text((x1 + x2) * 0.5, y_text, sig, ha='center', va='bottom', color=col, fontsize=10)
            
            # Increment level so next bracket goes higher
            level += 1


def plot_histogram(data, column, title, xlabel, output_dir, filename, hue=None):
    """
    Generates a Histogram. 
    If hue is provided, it overlaps distributions transparently.
    """
    plt.figure(figsize=(8, 6))
    
    # Check if we are splitting by category (hue)
    if hue:
        # Use MFA Blues/Reds for categories
        sns.histplot(
            data=data, x=column, hue=hue, kde=True, 
            element="step", fill=True, alpha=0.3, # Transparent overlap
            palette=get_experiment_palette(), 
            common_norm=False # Normalize each group independently
        )
    else:
        # Single global color
        mean_val = np.mean(data[column])
        sns.histplot(data=data, x=column, kde=True, color=MFA_COLORS['medium_blue'], 
                     edgecolor='white', alpha=0.8)
        plt.axvline(mean_val, color=MFA_COLORS['dark_red'], linestyle='--', lw=2, label=f'Mean: {mean_val:.2f}')
        plt.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count Density' if hue else 'Count')
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
    
def plot_boxplot_comparison(data, x_col, y_col, title, ylabel, output_dir=None, filename=None, ax=None, order=None, hue=None, box_pairs=None):
    """
    Generates a boxplot. 
    Can be grouped (hue) to show phenotypes within categories.
    Now supports statistical brackets via 'box_pairs'.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        is_standalone = True
    else:
        is_standalone = False

    # Determine Hue and Palette automatically
    # If hue is not explicitly passed, use the X column (simple 1-level plot)
    plot_hue = hue if hue else x_col
    
    # Select Palette based on what we are coloring by
    if plot_hue == 'Phenotype_Category':
        palette = PHENOTYPE_PALETTE
    else:
        palette = get_experiment_palette()

    # Determine Orders explicitly to ensure brackets align
    if order is None: 
        order = sorted(data[x_col].unique())
    
    hue_order = None
    if hue:
        # Standard Phenotype Order for consistency
        possible_order = ['Empty', 'Lumenal Actin', 'Sparse', 'Patchy', 'Uniform']
        present_hues = data[hue].unique()
        # Create order based on what is actually present in data
        hue_order = [h for h in possible_order if h in present_hues]
        # Add any unexpected ones at the end
        for h in present_hues:
            if h not in hue_order: 
                hue_order.append(h)

    # Draw Boxplot
    # dodge=True ensures the bars sit side-by-side
    sns.boxplot(data=data, x=x_col, y=y_col, hue=plot_hue, ax=ax, 
                palette=palette, order=order, hue_order=hue_order,
                boxprops=dict(alpha=0.8), showfliers=False, dodge=True)
    
    # Draw Stripplot (Dots) - must enable dodge to align with boxes
    sns.stripplot(data=data, x=x_col, y=y_col, hue=plot_hue, 
                  color='black', alpha=0.3, jitter=True, size=2, 
                  ax=ax, order=order, hue_order=hue_order, dodge=True, legend=False)

    # --- NEW: Add Statistical Brackets if requested ---
    if box_pairs and hue and hue_order:
        add_grouped_significance_brackets(ax, data, x_col, y_col, hue, box_pairs, order, hue_order)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('') # Clear X label as it's often redundant with tick labels
    
    # Improve Legend: Only show it if we are actually grouping
    if hue:
        ax.legend(title=hue.replace('_', ' '), bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    else:
        if ax.get_legend(): ax.get_legend().remove()

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

def plot_correlation_heatmap(corr_matrix, output_dir, filename_suffix=""):
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
    
    plt.title(f'Spearman Correlation Matrix {filename_suffix}', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Use the suffix in the filename
    save_plot(f"Spearman_Correlation_Heatmap{filename_suffix}.png", output_dir)