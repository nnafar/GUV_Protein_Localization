# -*- coding: utf-8 -*-
"""
MASTER PIPELINE
Orchestrates loading, processing, and analyzing GUV batch data.

CHANGES FROM PREVIOUS VERSION:
-------------------------------
1. CATEGORIES now contains all 4 experimental conditions.
2. Each condition has its own phenotyping logic (see analysis_phenotype.py).
"""

import os
import file_handling
import analysis_size
import analysis_phenotype
import analysis_statistics
import plotting

# ============================================================
#  CONFIGURATION — Edit these values to match your experiment
# ============================================================

# The top-level folder that contains all your experiment sub-folders.
# Each sub-folder should be named like: 260208_BranchedCortex_1
ROOT_PATH = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\Output"

# The four condition names, exactly as they appear in your folder names.
# These are case-insensitive (so "BranchedCortex" matches "branchedcortex").
CATEGORIES = ['BranchedCortex', 'LinearCortex', 'Factin', 'Empty']

# ============================================================


def main():
    # ----- Step 1: Create the output folder -----
    results_dir = file_handling.create_output_folder(ROOT_PATH, "Batch_Analysis_Results")
    print(f"Results will be saved to: {results_dir}\n")

    # ----- Step 2: Load all data -----
    # file_handling.py will scan ROOT_PATH, find all Analysis_Results.csv files,
    # assign a Category to each row based on the folder name, and combine everything
    # into one big table (called a DataFrame, or "df").
    df = file_handling.load_and_process_data(ROOT_PATH, CATEGORIES)

    if df is None:
        print("No data found. Please check ROOT_PATH and folder names.")
        return

    # Save the combined dataset for your records
    master_csv_path = os.path.join(results_dir, "Master_Dataset_Combined.csv")
    df.to_csv(master_csv_path, index=False)
    print(f"Master CSV saved: {master_csv_path}\n")

    # ----- Step 3: Run analysis modules -----

    # A. Size Distribution (violin + scatter + box plot across all 4 conditions)
    analysis_size.run_size_analysis(df, results_dir)

    # B. Phenotype Classification + Visualizations
    #    Each condition is classified differently — see analysis_phenotype.py
    analysis_phenotype.run_phenotype_analysis(df, results_dir)

    # C. Statistical Analysis (correlations + within-condition phenotype comparisons)
    analysis_statistics.run_statistics(df, results_dir)

    print("\n--- Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()