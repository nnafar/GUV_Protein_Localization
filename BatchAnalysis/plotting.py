# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 16:30:31 2026

@author: nnafar
"""

import os
import matplotlib
matplotlib.use('Agg') # Add this before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_plot(filename, output_dir):
    """Helper to save and close plots."""
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")

def plot_histogram(data, column, title, xlabel, output_dir, filename, color='teal', bins='rice'):
    """Generates a standard Histogram with KDE."""
    plt.figure(figsize=(8, 6), dpi=125)
    
    mean_val = np.mean(data[column])
    
    sns.histplot(data=data, x=column, bins=bins, kde=True, color=color, edgecolor='black', alpha=0.6)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_plot(filename, output_dir)

def plot_boxplot_comparison(data, x_col, y_col, title, ylabel, output_dir, filename, hue=None):
    """Generates a Boxplot with Strip (jitter) points overlay."""
    plt.figure(figsize=(10, 6), dpi=125)
    
    # Assign 'x' to 'hue' if no specific hue is provided to satisfy FutureWarnings
    plot_hue = hue if hue is not None else x_col
    
    # Plot Boxplot (legend=False prevents duplicate legend entries)
    sns.boxplot(data=data, x=x_col, y=y_col, hue=plot_hue, palette="Set2", showfliers=False, legend=False)
    
    # Plot Strip points (jitter)
    sns.stripplot(data=data, x=x_col, y=y_col, hue=hue, color='black', size=3, alpha=0.4, dodge=True)
    
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Clean up legend if stripplot added one
    if plt.gca().get_legend():
        plt.gca().get_legend().remove()
    
    save_plot(filename, output_dir)

def plot_scatter(data, x_col, y_col, hue_col, title, xlabel, ylabel, output_dir, filename):
    """Generates a Scatter plot colored by category."""
    plt.figure(figsize=(9, 7), dpi=125)
    
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, style=hue_col, s=60, alpha=0.7, palette="viridis")
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_plot(filename, output_dir)
    

def plot_phenotype_composition(data, category_col, phenotype_col, output_dir, filename):
    """
    Generates a 100% Stacked Bar Plot showing the composition of phenotypes
    within each experimental category.
    """
    # 1. Calculate Counts and Normalize to Percentage
    counts = data.groupby([category_col, phenotype_col]).size().unstack(fill_value=0)
    
    # Convert to percentages (row-wise normalization)
    percents = counts.div(counts.sum(axis=1), axis=0) * 100
    
    # 2. Plot
    plt.figure(figsize=(10, 6), dpi=125)
    
    # Use pandas plotting for the stacked structure
    ax = percents.plot(kind='bar', stacked=True, colormap='viridis', alpha=0.85, edgecolor='black', rot=45)
    
    plt.title('Phenotypic Composition by Category', fontsize=14)
    plt.xlabel('Experimental Category', fontsize=12)
    plt.ylabel('Percentage of Vesicles (%)', fontsize=12)
    plt.legend(title='Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # Save using the figure object
    fig = ax.get_figure()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Plot saved: {filename}")
    
    
def plot_violin_comparison(data, x_col, y_col, title, ylabel, output_dir, filename):
    """Generates a Violin plot to show density distribution."""
    plt.figure(figsize=(10, 6), dpi=125)
    
    # Inner='quartile' draws dashed lines for median and quartiles
    # Set hue=x_col and legend=False to fix the warning
    sns.violinplot(data=data, x=x_col, y=y_col, hue=x_col, legend=False, palette="Set2", inner="quartile", alpha=0.6)
    
    # Optional: Add strip plot on top for individual points (if data isn't too huge)
    sns.stripplot(data=data, x=x_col, y=y_col, color='black', size=2, alpha=0.3, jitter=True)
    
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    save_plot(filename, output_dir)
    

def plot_boxplot_comparison(data, x_col, y_col, title, ylabel, output_dir=None, filename=None, ax=None):
    """
    Generates a boxplot comparing a numerical variable across categories.
    Can draw on an existing axis 'ax' if provided, or create a new figure.
    """
    # Determine if we are drawing on an existing subplot or creating a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        is_standalone = True
    else:
        # We are drawing onto an existing subplot passed from outside
        is_standalone = False

    # Create Boxplot on the specified axis
    # Added 'hue' to avoid future warning, set legend=False
    sns.boxplot(data=data, x=x_col, y=y_col, hue=x_col, legend=False, ax=ax, palette='Set2')
    
    # Overlay strip plot for individual data points (optional, but nice)
    sns.stripplot(data=data, x=x_col, y=y_col, color='black', alpha=0.3, jitter=True, ax=ax)

    # Customize the specific axis
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(x_col, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Only save automatically if it's a standalone plot AND filename is provided
    if is_standalone and filename and output_dir:
         save_plot(filename, output_dir)