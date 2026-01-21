# -*- coding: utf-8 -*-
"""
MASTER PIPELINE
Orchestrates loading, processing, and analyzing GUV batch data.
"""

import os
import file_handling
import analysis_size
import analysis_phenotype
import analysis_statistics
import plotting

# ---------------- CONFIGURATION ----------------
# Use raw string (r"...") for Windows paths to avoid escape character issues
ROOT_PATH = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\Output"
CATEGORIES = ['BranchedCortex', 'LinearCortex', 'Empty', 'Actin']
# -----------------------------------------------

def main():
    # 1. Setup Output
    results_dir = file_handling.create_output_folder(ROOT_PATH, "Batch_Analysis_Results")
    print(f"Results will be saved to: {results_dir}")

    # 2. Load Data
    df = file_handling.load_and_process_data(ROOT_PATH, CATEGORIES)
    
    if df is not None:
        # Save the combined Master CSV for record keeping
        master_csv_path = os.path.join(results_dir, "Master_Dataset_Combined.csv")
        df.to_csv(master_csv_path, index=False)
        print(f"Master CSV saved: {master_csv_path}")

        # 3. Run Analysis Modules
        # A. Size Distribution
        analysis_size.run_size_analysis(df, results_dir)
        
        # B. Phenotype Characterization (Density, Thickness, Correlations)
        # This module now internally handles Categorization and the Stacked Bar Plot
        analysis_phenotype.run_phenotype_analysis(df, results_dir)
        
        # C. Batch-to-Batch Comparison (Drill down)
        #analysis_phenotype.run_batch_comparison(df, results_dir)
        
        # D. Statistical Analysis
        analysis_statistics.run_statistics(df, results_dir)
        
        print("\n--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()