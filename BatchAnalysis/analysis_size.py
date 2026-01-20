# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 16:31:27 2026

@author: nnafar
"""

import plotting

def run_size_analysis(df, output_dir):
    """
    Analyzes and plots the size distribution (Refined Radius).
    """
    print("\n--- Running Size Distribution Analysis ---")
    
    # 1. Global Size Distribution
    plotting.plot_histogram(
        data=df,
        column='Refined Radius (um)',
        title='Global Size Distribution (All Categories)',
        xlabel='Refined Radius (µm)',
        output_dir=output_dir,
        filename='Global_Size_Distribution.png',
        color='steelblue'
    )
    
    # 2. Size Distribution by Category
    plotting.plot_boxplot_comparison(
        data=df,
        x_col='Category',
        y_col='Refined Radius (um)',
        title='Size Comparison by Category',
        ylabel='Radius (µm)',
        output_dir=output_dir,
        filename='Category_Size_Comparison.png'
    )