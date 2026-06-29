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
import analysis_distribution_shape
import analysis_pca
import analysis_batch
import analysis_comparison
import figure_assembly
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

    # ─── Step 1: Create the output folder ────────────────────────────────────
    results_dir = file_handling.create_output_folder(
        ROOT_PATH, "Batch_Analysis_Results"
    )
    print(f"Results will be saved to: {results_dir}\n")

    # ─── Step 2: Load all data ────────────────────────────────────────────────
    df = file_handling.load_and_process_data(ROOT_PATH, CATEGORIES)

    if df is None:
        print("No data found. Please check ROOT_PATH and folder names.")
        return

    # Save the combined dataset so you can open it in Excel or re-use it later
    master_csv_path = os.path.join(results_dir, "Master_Dataset_Combined.csv")
    df.to_csv(master_csv_path, index=False)
    print(f"Master CSV saved: {master_csv_path}\n")

    # ─── Step 3: Size Distribution ───────────────────────────────────────────
    # Violin + scatter + box plot of vesicle radius for all 4 conditions.
    # Also saves descriptive statistics (mean, std, SEM) to CSV.
    analysis_size.run_size_analysis(df, results_dir)

    # ─── Step 4: Phenotype Classification ────────────────────────────────────
    # Classifies each vesicle (SPARSE / PATCHY / CONTINUOUS / etc.) and then
    # quantifies the SHAPE of every per-vesicle metric distribution
    # (skewness, kurtosis) at the population, cortex-only, and per-phenotype
    # levels. Outputs traceable shape moments cited in the Results section.
    analysis_phenotype.run_phenotype_analysis(df, results_dir)
    analysis_distribution_shape.run_distribution_shape_analysis(df, results_dir)
    
    # ─── Step 5: Statistical Analysis ────────────────────────────────────────
    # Spearman correlations and within-condition phenotype comparisons
    # (Mann-Whitney U + Cliff's Delta effect sizes).
    analysis_statistics.run_comparison_analysis(df, results_dir)
    
    
    # ─── Step 5.5: PCA on cortex metrics ─────────────────────────────────────
    # Collapses the three correlated cortex metrics (A localization, Gini, t_cortex)
    # onto an orthogonal basis. PC1 = cortex maturity axis, PC2 = architectural
    # identity axis. Statistical tests are then run on PC scores instead of on
    # the three correlated metrics, eliminating the multiplicity problem.
    #
    # MUST come after Step 4 (phenotype classification) because the analysis is
    # restricted to the cortex-forming subset (Factin SHELL, BranchedCortex
    # CONTINUOUS, LinearCortex CONTINUOUS).
    #
    # WHAT DOES THIS STEP PRODUCE?
    # In results_dir/PCA_Analysis/:
    #   PCA_Loadings.csv               — how each PC is built from the metrics
    #   PCA_Explained_Variance.csv     — fraction of variance per PC
    #   PCA_Vesicle_Scores.csv         — per-vesicle PC1/PC2/PC3 scores
    #   PCA_Statistical_Tests.csv      — Mann-Whitney + Cliff's Delta on PCs
    #   PCA_Bootstrap_Sensitivity.csv  — loading stability over 200 resamples
    #   Plot_PCA_Loadings.png          — bar chart of loadings
    #   Plot_PCA_Scree.png             — explained variance per PC
    #   Plot_PCA_Scatter.png           — PC1 vs PC2, coloured by condition
    analysis_pca.run_pca_analysis(df, results_dir)
    
    # ─── Step 6: Batch-to-Batch Variability ──────────────────────────────────
    # Checks whether batches within each condition are internally consistent.
    # Dots in the actin metrics plot are now coloured by phenotype so you can
    # see whether batch differences are driven by phenotype composition shifts
    # rather than genuine preparation variability.
    #
    # WHAT DOES THIS STEP PRODUCE?
    # In results_dir/Batch_Analysis/:
    #   Batch_Statistics_Summary.csv           — mean/std/N per batch per metric
    #   Batch_Consistency_Tests.csv            — Kruskal-Wallis p-values
    #   Batch_Size_Distribution.png            — radius across batches
    #   Batch_Phenotype_Composition.png        — % of each phenotype per batch
    #   Batch_Actin_Metrics_All_Conditions.png — actin metrics, dots by phenotype
    analysis_batch.run_batch_analysis(df, results_dir)

    # ─── Step 7: Cross-Condition Comparison ──────────────────────────────────
    # MUST come after Step 4 (phenotype classification), because the
    # phenotype-stratified comparison (Plot F) requires 'Phenotype_Category'.
    #
    # WHAT DOES THIS STEP PRODUCE?
    # In results_dir/Comparison_Analysis/:
    #   Comparison_Summary_Statistics.csv            — median/IQR per condition
    #   Comparison_Statistical_Tests.csv             — p-values, Cliff's Delta
    #   Plot_Radius_AllConditions.png                — radius: all 4 conditions
    #   Plot_Cortex_vs_Empty_Radius.png              — radius: cortex vs empty
    #   Plot_Actin_ThreeWay.png                      — Factin/Branched/Linear
    #   Plot_LumenRetention.png                      — A Lumen/Bg comparison
    #   Plot_ShapeMetrics_AllConditions.png          — deformability + bumpiness
    #   Plot_Phenotype_Stratified_Continuous.png     — CONTINUOUS-only comparison
    analysis_comparison.run_comparison_analysis(df, results_dir)

    # ─── Step 8: Figure Assembly ─────────────────────────────────────────────
    # Composes individual plots into complete thesis figures (main + supplementary).
    # MUST come last — requires all analysis steps to be complete and all
    # plot columns (including Phenotype_Category) to be present in df.
    #
    # WHAT DOES THIS STEP PRODUCE?
    # In results_dir/Figures/:
    #   Main_Figure_1_Phenotype_Composition.pdf
    #   Main_Figure_2_Characteristics_Matrix.pdf
    #   Main_Figure_3_Actin_Comparison.pdf
    #   Main_Figure_4_Phenotype_Stratified.pdf
    #   Supp_Figure_S1_Radius.pdf
    #   Supp_Figure_S2_Batch_QC.pdf
    #   Supp_Figure_S3_Batch_Actin_Metrics.pdf
    #   Supp_Figure_S4_Correlation_Matrices.pdf
    #   Supp_Figure_S5_Lumen_Retention.pdf
    figure_assembly.run_figure_assembly(df, results_dir)

    print("\n--- Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()