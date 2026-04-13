# -*- coding: utf-8 -*-
"""
SIZE ANALYSIS MODULE
Analyzes the physical dimensions (radius) of vesicles across all 4 conditions.

CHANGES FROM PREVIOUS VERSION:
-------------------------------
1. The histogram has been replaced with a combined violin + scatter + box plot.
   This shows:
     - The full distribution shape     (violin — wide where many vesicles)
     - Individual vesicle data points  (scatter / strip plot)
     - Median and quartile lines       (box plot overlaid on the violin)

2. The violin plot is now the only size comparison plot
   (previously there was a separate violin — that's now integrated).
"""

import pandas as pd
import os
import plotting


def save_size_statistics(df, output_dir):
    """
    Calculates descriptive statistics (mean, std, SEM, count) for vesicle radius
    grouped by condition (Category), and saves them to a CSV file.

    Think of this like calculating the average height of students
    in different classes and writing it down.
    """
    print("  -> Calculating Size Statistics...")

    stats = (
        df.groupby('Category')['Refined Radius (um)']
        .agg(['mean', 'std', 'sem', 'count'])
        .rename(columns={
            'mean':  'Mean_Radius_um',
            'std':   'Std_Dev_um',
            'sem':   'SEM_um',
            'count': 'N_Vesicles',
        })
        .reset_index()
    )

    path = os.path.join(output_dir, "Size_Statistics_Summary.csv")
    stats.to_csv(path, index=False)
    print(f"  -> Size statistics saved: {path}")


def run_size_analysis(df, output_dir):
    """
    Generates the size distribution plot and saves size statistics.

    Parameters
    ----------
    df         : pandas DataFrame — must contain 'Refined Radius (um)' and 'Category'
    output_dir : str              — folder where output files are saved
    """
    print("\n--- Running Size Distribution Analysis ---")

    # Combined violin + scatter + box plot for all 4 conditions
    plotting.plot_violin_scatter_box(
        data=df,
        x_col='Category',
        y_col='Refined Radius (um)',
        title='Vesicle Size Distribution by Condition',
        ylabel='Refined Radius (µm)',
        output_dir=output_dir,
        filename='Size_Distribution_Violin_Scatter_Box.png',
    )

    # Save descriptive statistics to CSV
    save_size_statistics(df, output_dir)