# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import plotting

# ----------------- PHYSICAL THRESHOLDS -----------------
LUMEN_LOC_LIMIT = 0.40    # Below this is strictly No Localization
STRUCTURED_LOC_MIN = 0.79 # Above this is the 'Dense/Thin' cortex
EMPTY_RHO_LIMIT = 2.0     # Baseline for completely empty signal
# -------------------------------------------------------

def categorize_vesicles(df):
    """
    Classifies vesicles based on your observed trends:
    1. Empty/Lumenal (< 0.40 Loc)
    2. Fuzzy Patches (0.40 - 0.79 Loc)
    3. Structured Cortex (> 0.79 Loc)
    """
    def classify(row):
        loc = row['A localization']
        rho = row['rho_actin']
        
        if pd.isna(loc) or pd.isna(rho):
            return "Excluded"

        # 1. No Localization Family
        if loc < LUMEN_LOC_LIMIT:
            if rho < EMPTY_RHO_LIMIT:
                return "Empty"
            else:
                return "Lumenal Actin"
            
        # 2. Transition / Fuzzy Patches
        if LUMEN_LOC_LIMIT <= loc < STRUCTURED_LOC_MIN:
            return "Thick Fuzzy Patch"
            
        # 3. Structured Cortical Family
        if loc >= STRUCTURED_LOC_MIN:
            return "Thin & Dense Cortex"
            
        return "Unclassified"

    df['Phenotype_Category'] = df.apply(classify, axis=1)
    return df, {'structured_min': STRUCTURED_LOC_MIN, 'empty_rho': EMPTY_RHO_LIMIT}

def run_phenotype_analysis(df, output_dir):
    """Analyzes phenotypes using the corrected 0.40/0.79 boundaries."""
    print("\n--- Running Phenotype Analysis (Corrected Thresholds) ---")

    df, stats = categorize_vesicles(df)
    
    # Save the categorization results
    log_cols = ['Date', 'Name', 'Vesicle id', 'Phenotype_Category', 'A localization', 't_cortex', 'rho_actin', 'Comment']
    log_path = os.path.join(output_dir, "Phenotype_Categorization_Log.csv")
    df[log_cols].to_csv(log_path, index=False)
    print(f"  -> Categorization Log saved: {log_path}")

    # Plot results
    plotting.plot_phenotype_composition(
        data=df,
        category_col='Category',
        phenotype_col='Phenotype_Category',
        output_dir=output_dir,
        filename='Phenotype_Composition_Stacked.png'
    )