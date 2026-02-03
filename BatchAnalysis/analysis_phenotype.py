# -*- coding: utf-8 -*-
"""
PHENOTYPE ANALYSIS MODULE
Handles the classification logic and orchestration of phenotype analysis.
"""

import pandas as pd
import numpy as np
import os
import plotting  # Delegates all visualization to this module

# =============================================================================
# 1. LOGIC FUNCTIONS
# =============================================================================

def categorize_vesicles(df):
    """
    Classifies vesicles based on defined thresholds.
    MODIFIES THE DATAFRAME IN-PLACE.
    """
    def classify(row):
        # 1. EXCLUDED
        if pd.isna(row['t_cortex']):
            return "EXCLUDED"
            
        loc = row['A localization']
        lumen_ratio = row['A Lumen/Bg']
        gini = row['Gini_Index']
        
        # 2 & 3. ZERO LOCALIZATION
        if loc <= 0.0:
            if lumen_ratio <= 2.0:
                return "EMPTY"
            else:
                return "LUMENAL"
        
        # 4, 5 & 6. POSITIVE LOCALIZATION
        if loc <= 0.65:
            return "SPARSE"
        else:
            if gini > 0.45:
                return "PATCHY"
            else:
                return "CONTINUOUS"

    # IN-PLACE Update (No copy)
    df['Phenotype_Category'] = df.apply(classify, axis=1)
    return df

# =============================================================================
# 2. MAIN EXECUTION
# =============================================================================

def run_phenotype_analysis(df, output_dir):
    print("\n--- Running Phenotype & Cortex Analysis (Category Split) ---")
    
    # 0. Setup
    plotting.set_paper_style()
    
    # 1. Run Classification Logic
    categorize_vesicles(df)
    
    # 1a. Save classification csv
    detailed_path = os.path.join(output_dir, "Vesicle_Phenotype_Assignments.csv")
    df.to_csv(detailed_path, index=False)
    print(f"      -> Detailed Assignments saved: {detailed_path}")
    
    # 2. Visualizations
    
    # A. Stacked Bar Chart
    plotting.plot_phenotype_composition(
        df, 'Category', 'Phenotype_Category', output_dir, 'Phenotype_Composition_Stacked.png'
    )
    
    # B. The Matrix Boxplot with Stats
    plotting.generate_category_panel(df, output_dir)
    
    # C. Split Maps & Pair Plots
    unique_cats = df['Category'].unique()
    for cat in unique_cats:
        cat_df = df[df['Category'] == cat]
        if not cat_df.empty:
            # 5D Map
            plotting.plot_cortex_map(cat_df, cat, output_dir)
            
            # Pair Plot (Restored)
            plotting.plot_pairplot(cat_df, output_dir, filename_suffix=f"_{cat}")

    print("      -> Phenotype Analysis Complete.")