# -*- coding: utf-8 -*-
"""
SIZE ANALYSIS MODULE
Analyzes the physical dimensions of vesicles.
"""
import pandas as pd
import os
import plotting

def save_size_statistics(df, output_dir):
    """
    Calculates Mean, Std, SEM, and Count for vesicle size by Category.
    Saves to CSV.
    """
    print("  -> Calculating Size Statistics...")
    
    # Group by Category and calculate stats for Radius
    stats = df.groupby('Category')['Refined Radius (um)'].agg(['mean', 'std', 'sem', 'count'])
    
    # Rename columns for clarity
    stats.columns = ['Mean_Radius_um', 'Std_Dev_um', 'SEM_um', 'N_Vesicles']
    stats = stats.reset_index()
    
    # Save
    path = os.path.join(output_dir, "Size_Statistics_Summary.csv")
    stats.to_csv(path, index=False)
    print(f"  -> Stats saved: {path}")

def run_size_analysis(df, output_dir):
    """
    Triggers the generation of size-related visualizations and stats.
    """
    print("\n--- Running Size Distribution Analysis ---")
    
    # 1. Comparison Histogram (Transparent Overlap)
    # Uses 'hue' to separate Branched/Linear on the same plot
    plotting.plot_histogram(
        data=df,
        column='Refined Radius (um)',
        title='Size Distribution by Category',
        xlabel='Refined Radius (µm)',
        output_dir=output_dir,
        filename='Global_Size_Distribution.png', # Filename kept, content changed
        hue='Category'
    )

    # 2. Violin Plot
    plotting.plot_violin_comparison(
        data=df,
        x_col='Category',
        y_col='Refined Radius (um)',
        title='Size Density Distribution by Category (Violin)',
        ylabel='Radius (µm)',
        output_dir=output_dir,
        filename='Category_Size_Comparison_Violin.png'
    )
    
    # 3. Save Statistics
    save_size_statistics(df, output_dir)