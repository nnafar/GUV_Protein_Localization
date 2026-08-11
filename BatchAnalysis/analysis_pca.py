# -*- coding: utf-8 -*-
"""
PRINCIPAL COMPONENT ANALYSIS MODULE
=====================================

PURPOSE
-------
This module collapses the three correlated cortex metrics
(A localization, Gini_Index, t_cortex) onto an orthogonal basis
so that downstream statistical tests are run on independent axes.

WHY?
----
Spearman correlations between A localization, Gini_Index, and t_cortex
within each actin condition are ρ = 0.74–0.93 (see analysis_statistics).
Reporting Mann-Whitney + Cliff's Delta on each of the three metrics in
parallel therefore overstates how many independent observations support
any between-condition contrast.

PCA rewrites the three correlated metrics as three uncorrelated
"principal components" (PCs):
    PC1  -- the cortex MATURITY axis
            (no-cortex → fully-formed; all three metrics increase together)
    PC2  -- the cortex IDENTITY axis
            (the residual variation perpendicular to PC1; this is where
             the architectural difference between branched and linear
             cortices is expected to live)
    PC3  -- residual noise (usually <10% of total variance)

Each GUV is then described by three numbers (PC1, PC2, PC3) that are,
by construction, mathematically independent of each other.

DESIGN DECISIONS
----------------
1. POOLED basis across the three cortex-forming subsets
   (Factin SHELL + BranchedCortex CONTINUOUS + LinearCortex CONTINUOUS).
   This keeps PC scores directly comparable across conditions — required
   for the between-condition tests that are the whole point of the analysis.

2. Restricted to CORTEX-FORMING vesicles only.
   If the full population were used, PC1 would be dominated by the
   zero/non-zero split (cortex present vs absent), pushing the
   architectural-identity axis to PC3 and beyond. Restricting to the
   cortex-forming subset isolates the architectural question.

3. Z-scoring is computed on the POOLED subset (not per-condition).
   This is essential — per-condition z-scoring would erase the between-
   condition differences before they reach the PCA.

4. SIGN FIX: PC1 is forced to have a positive loading on z(A localization)
   so that "high PC1" always means "more mature cortex" regardless of
   numerical seed. Without this, PCA sign is arbitrary and can flip
   between runs, which breaks downstream interpretation.

OUTPUTS  (all saved to <output_dir>/PCA_Analysis/)
----------------------------------------------------
CSV files:
    PCA_Loadings.csv                — how each PC is built from the metrics
    PCA_Explained_Variance.csv      — fraction of variance per PC
    PCA_Vesicle_Scores.csv          — per-vesicle PC1, PC2, PC3 scores
    PCA_Statistical_Tests.csv       — Mann-Whitney + Cliff's Delta on PC scores
    PCA_Bootstrap_Sensitivity.csv   — loading stability over 200 resamples

Plots (saved as PDFs):
    Plot_PCA_Loadings.pdf   — bar chart: loading of each metric on each PC
    Plot_PCA_Scree.pdf      — explained variance per PC
    Plot_PCA_Scatter.pdf    — PC1 vs PC2, coloured by condition,
                              with marginal density curves
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")        # save to file; don't try to open a screen window
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotting

# =============================================================================
# STANDALONE CONFIGURATION
# Only used when you run: python analysis_pca.py directly.
# Ignored when imported by master_pipeline.py.
# =============================================================================

ROOT_PATH = "H:/ProteinLocalization/Output"
STANDALONE_OUTPUT_DIR = os.path.join(ROOT_PATH, "Batch_Analysis_Results")
STANDALONE_CSV_PATH = os.path.join(STANDALONE_OUTPUT_DIR, "Vesicle_Phenotype_Assignments.csv")

# =============================================================================
# CONSTANTS
# =============================================================================

# The three metrics that go into the PCA.
# Order matters here -- loadings will be reported in this order.
PCA_METRICS = ['A localization', 'Gini_Index', 't_cortex']

# Human-readable axis labels for plots and CSVs.
METRIC_LABELS = {
    'A localization': 'Localization\nScore',
    'Gini_Index':     'Gini\nIndex',
    't_cortex':       r'Cortex Thickness' '\n' r'($\mu$m)'
}

# The cortex-forming subset is defined as:
#   Factin         + Phenotype_Category == SHELL
#   BranchedCortex + Phenotype_Category == CONTINUOUS
#   LinearCortex   + Phenotype_Category == CONTINUOUS
#
# This is exactly the subset used in the phenotype-stratified
# comparison (Analysis Group E in analysis_comparison.py).
CORTEX_FORMING_FILTER = {
    'Factin':         {'Shell'},
    'BranchedCortex': {'Continuous'},
    'LinearCortex':   {'Continuous'},
}

# Bootstrap parameters for the sensitivity check.
N_BOOTSTRAPS    = 200
BOOTSTRAP_SEED  = 42


# =============================================================================
# SHARED HELPERS  (kept local; duplicated from analysis_comparison.py
# rather than imported, to avoid creating a cross-module dependency)
# =============================================================================

def _cliffs_delta(a, b):
    """
    Calculates Cliff's Delta -- the practical size of a difference.

    The p-value tells you "is this difference real?"
    Cliff's Delta tells you "how big is it in practice?"
      0.00 – 0.15  → Negligible
      0.15 – 0.33  → Small
      0.33 – 0.47  → Medium
      0.47+        → Large

    Returns the ABSOLUTE value (between 0.0 and 1.0).
    """
    a, b = np.sort(a), np.sort(b)
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0

    j = k = more = less = 0
    for x in a:
        while j < n and b[j] < x:
            j += 1
        while k < n and b[k] <= x:
            k += 1
        more += j
        less += (n - k)

    return abs((more - less) / (m * n))


def _effect_label(d):
    """Converts a Cliff's Delta value to a human-readable size label."""
    if d < 0.147: return "Negligible"
    if d < 0.330: return "Small"
    if d < 0.474: return "Medium"
    return "Large"


def _stars(p):
    """Standard significance star encoding."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# =============================================================================
# STEP 1 -- BUILD THE CORTEX-FORMING SUBSET
# =============================================================================

def build_cortex_forming_subset(df):
    """
    Filters the master DataFrame down to the three cortex-forming subsets:
      Factin SHELL, BranchedCortex CONTINUOUS, LinearCortex CONTINUOUS.

    Drops any vesicle with a missing value in any of the three PCA metrics
    (PCA cannot handle NaNs and dropping is safer than imputing here --
    a NaN in t_cortex means membrane detection failed, which is a
    qualitative signal we don't want to overwrite with a guess).

    Parameters
    ----------
    df : master DataFrame after analysis_phenotype has run.
         Must contain 'Category', 'Phenotype_Category', and the PCA metrics.

    Returns
    -------
    subset : DataFrame with the same columns as df, restricted to the
             three cortex-forming subsets and complete on PCA metrics.
    """
    if 'Phenotype_Category' not in df.columns:
        raise RuntimeError(
            "'Phenotype_Category' not found -- "
            "run analysis_phenotype before analysis_pca."
        )

    # Build the filter mask: keep rows whose (Category, Phenotype) pair
    # appears in CORTEX_FORMING_FILTER.
    keep = pd.Series(False, index=df.index)
    for category, phenotypes in CORTEX_FORMING_FILTER.items():
        mask = (df['Category'] == category) & (df['Phenotype_Category'].isin(phenotypes))
        keep |= mask

    subset = df.loc[keep].copy()

    # Drop rows with NaN in any PCA metric.
    before = len(subset)
    subset = subset.dropna(subset=PCA_METRICS)
    dropped = before - len(subset)

    if dropped > 0:
        print(f"      -> Dropped {dropped} vesicles with NaN in a PCA metric")

    # Print the subset composition.
    print("      -> Cortex-forming subset:")
    for category in plotting.CONDITION_ORDER:
        if category not in CORTEX_FORMING_FILTER:
            continue
        n = (subset['Category'] == category).sum()
        print(f"         {plotting.CONDITION_LABELS.get(category, category):>15} : "
              f"{n} vesicles")
    print(f"         {'TOTAL':>15} : {len(subset)} vesicles")

    return subset


# =============================================================================
# STEP 2 -- FIT THE PCA
# =============================================================================

def fit_pca(subset, output_dir):
    """
    Fits a 3-component PCA on the pooled, z-scored cortex-forming subset.

    Sign-fixes PC1 so that its loading on A localization is positive
    (the natural "more cortex" direction). This keeps interpretation
    stable across runs regardless of numerical seed.

    Parameters
    ----------
    subset     : DataFrame from build_cortex_forming_subset().
                 Must contain the three PCA metrics.
    output_dir : folder where Loadings + ExplainedVariance + Scores CSVs go.

    Returns
    -------
    scores_df : original subset DataFrame with three new columns
                ('PC1', 'PC2', 'PC3') containing the per-vesicle scores.
    loadings  : (3, 3) numpy array. Row i, column j = loading of
                metric j on PCi.
    var_ratio : length-3 numpy array of explained-variance fractions.
    """
    print("  -> Fitting PCA on pooled cortex-forming subset...")

    # ---- 1. Extract and z-score the three metrics ----------------------
    X = subset[PCA_METRICS].values
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    # ---- 2. Fit PCA ----------------------------------------------------
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X_z)
    loadings = pca.components_          # shape (3 PCs, 3 metrics)
    var_ratio = pca.explained_variance_ratio_

    # ---- 3. Sign-fix PC1 ----------------------------------------------
    idx_aloc = PCA_METRICS.index('A localization')
    if loadings[0, idx_aloc] < 0:
        loadings[0, :] *= -1
        scores[:, 0]   *= -1
        print("      -> PC1 sign flipped to enforce A_loc > 0 convention")

    idx_tcortex = PCA_METRICS.index('t_cortex')
    if loadings[1, idx_tcortex] < 0:
        loadings[1, :] *= -1
        scores[:, 1]   *= -1
        print("      -> PC2 sign flipped to enforce t_cortex > 0 convention")

    # ---- 4. Save loadings table ---------------------------------------
    loadings_df = pd.DataFrame(
        loadings,
        index   = ['PC1', 'PC2', 'PC3'],
        columns = [f"loading_on_z({METRIC_LABELS[m]})" for m in PCA_METRICS],
    )
    loadings_df.insert(0, 'Component', loadings_df.index)
    loadings_df = loadings_df.reset_index(drop=True)

    path = os.path.join(output_dir, "PCA_Loadings.csv")
    loadings_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    # ---- 5. Save explained variance -----------------------------------
    var_df = pd.DataFrame({
        'Component':                  ['PC1', 'PC2', 'PC3'],
        'Explained_Variance_Fraction': np.round(var_ratio, 4),
        'Explained_Variance_Percent':  np.round(var_ratio * 100, 2),
        'Cumulative_Percent':          np.round(np.cumsum(var_ratio) * 100, 2),
    })

    path = os.path.join(output_dir, "PCA_Explained_Variance.csv")
    var_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    # ---- 6. Attach scores back to the subset --------------------------
    scores_df = subset.copy()
    scores_df['PC1'] = np.round(scores[:, 0], 4)
    scores_df['PC2'] = np.round(scores[:, 1], 4)
    scores_df['PC3'] = np.round(scores[:, 2], 4)

    slim_cols = ['Date', 'Image', 'Vesicle id', 'Category', 'Batch_ID',
                 'Phenotype_Category', 'Refined Radius (um)'] + PCA_METRICS \
                + ['PC1', 'PC2', 'PC3']
    slim_cols = [c for c in slim_cols if c in scores_df.columns]

    path = os.path.join(output_dir, "PCA_Vesicle_Scores.csv")
    scores_df[slim_cols].to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    # ---- 7. Console summary -------------------------------------------
    print("\n      === PCA loadings (z-scored metrics) ===")
    print(loadings_df.to_string(index=False))
    print("\n      === Explained variance ===")
    print(var_df.to_string(index=False))
    print()

    return scores_df, loadings, var_ratio


# =============================================================================
# STEP 3 -- STATISTICAL TESTS ON PC SCORES
# =============================================================================

def run_pca_statistical_tests(scores_df, output_dir):
    """
    Runs Mann-Whitney U + Cliff's Delta on PC1, PC2, and PC3 between
    each pair of the three cortex-forming subsets.
    """
    print("  -> Running statistical tests on PC scores...")

    results = []

    groups = [
        ('Factin',         'Shell',      'F-actin SHELL'),
        ('BranchedCortex', 'Continuous', 'Branched CONTINUOUS'),
        ('LinearCortex',   'Continuous', 'Linear CONTINUOUS'),
    ]

    pcs = ['PC1', 'PC2', 'PC3']
    group_scores = {}
    for cat, pheno, label in groups:
        sub = scores_df[(scores_df['Category'] == cat) &
                        (scores_df['Phenotype_Category'] == pheno)]
        group_scores[label] = {pc: sub[pc].dropna().values for pc in pcs}

    labels = [label for _, _, label in groups]
    for pc in pcs:
        for i, label_a in enumerate(labels):
            for label_b in labels[i + 1:]:

                vals_a = group_scores[label_a][pc]
                vals_b = group_scores[label_b][pc]

                if len(vals_a) < 5 or len(vals_b) < 5:
                    continue

                try:
                    _, p_val = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
                except Exception as e:
                    print(f"      ! Test failed ({label_a} vs {label_b}, {pc}): {e}")
                    continue

                d_val = _cliffs_delta(vals_a, vals_b)

                results.append({
                    'PC':                pc,
                    'Comparison':        f"{label_a}  vs  {label_b}",
                    'N_A':               len(vals_a),
                    'Median_A':          round(float(np.median(vals_a)), 4),
                    'N_B':               len(vals_b),
                    'Median_B':          round(float(np.median(vals_b)), 4),
                    'Median_Difference': round(float(np.median(vals_a) - np.median(vals_b)), 4),
                    'p_value':           round(float(p_val), 6),
                    'Significance':      _stars(p_val),
                    'Cliffs_Delta':      round(float(d_val), 4),
                    'Effect_Size':       _effect_label(d_val),
                })

    if not results:
        print("      -> No valid comparisons.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values(['PC', 'Comparison'])

    path = os.path.join(output_dir, "PCA_Statistical_Tests.csv")
    results_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    print("\n      === PCA-based statistical tests ===")
    for pc, grp_df in results_df.groupby('PC'):
        print(f"\n      [{pc}]")
        for _, row in grp_df.iterrows():
            flag = "  *" if row['p_value'] < 0.05 else "  -"
            print(
                f"{flag}  {row['Comparison']:<45}  "
                f"p={row['p_value']:.4f} {row['Significance']:>3}  "
                f"d={row['Cliffs_Delta']:.3f} ({row['Effect_Size']})"
            )
    print()

    return results_df


# =============================================================================
# STEP 4 -- BOOTSTRAP SENSITIVITY CHECK
# =============================================================================

def run_bootstrap_sensitivity(subset, output_dir,
                              n_boots=N_BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    """
    Tests how stable the PCA loadings and explained variance are when
    the three cortex-forming subsets are equal-N resampled.
    """
    print(f"  -> Bootstrap sensitivity check ({n_boots} iterations, "
          f"equal-N resampling)...")

    rng = np.random.default_rng(seed)

    indices_per_condition = {}
    for cat in CORTEX_FORMING_FILTER.keys():
        indices_per_condition[cat] = subset.index[subset['Category'] == cat].to_numpy()

    n_min = min(len(v) for v in indices_per_condition.values())
    print(f"      -> Resampling N = {n_min} per condition each iteration")

    boot_loadings  = []
    boot_variances = []

    for b in range(n_boots):
        chosen = []
        for cat, idx_array in indices_per_condition.items():
            chosen.append(rng.choice(idx_array, size=n_min, replace=False))
        chosen_idx = np.concatenate(chosen)

        X = subset.loc[chosen_idx, PCA_METRICS].values

        scaler = StandardScaler()
        X_z = scaler.fit_transform(X)

        pca = PCA(n_components=3)
        pca.fit(X_z)
        loadings  = pca.components_.copy()
        var_ratio = pca.explained_variance_ratio_.copy()

        idx_aloc    = PCA_METRICS.index('A localization')
        idx_tcortex = PCA_METRICS.index('t_cortex')
        if loadings[0, idx_aloc] < 0:
            loadings[0, :] *= -1
        if loadings[1, idx_tcortex] < 0:
            loadings[1, :] *= -1

        boot_loadings.append(loadings)
        boot_variances.append(var_ratio)

    boot_loadings  = np.stack(boot_loadings)
    boot_variances = np.stack(boot_variances)

    rows = []
    for pc_idx, pc in enumerate(['PC1', 'PC2', 'PC3']):
        for m_idx, metric in enumerate(PCA_METRICS):
            values = boot_loadings[:, pc_idx, m_idx]
            rows.append({
                'Component':       pc,
                'Metric':          METRIC_LABELS[metric],
                'Quantity':        'Loading',
                'Mean':            round(float(values.mean()), 4),
                'Std_Dev':         round(float(values.std()),  4),
                'Pct2_5':          round(float(np.percentile(values, 2.5)),  4),
                'Pct97_5':         round(float(np.percentile(values, 97.5)), 4),
            })

    for pc_idx, pc in enumerate(['PC1', 'PC2', 'PC3']):
        values = boot_variances[:, pc_idx]
        rows.append({
            'Component':  pc,
            'Metric':     '(explained variance fraction)',
            'Quantity':   'Variance',
            'Mean':       round(float(values.mean()), 4),
            'Std_Dev':    round(float(values.std()),  4),
            'Pct2_5':     round(float(np.percentile(values, 2.5)),  4),
            'Pct97_5':    round(float(np.percentile(values, 97.5)), 4),
        })

    sensitivity_df = pd.DataFrame(rows)

    path = os.path.join(output_dir, "PCA_Bootstrap_Sensitivity.csv")
    sensitivity_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    print("\n      === Bootstrap loadings (mean ± std over resamples) ===")
    print(sensitivity_df.to_string(index=False))
    print()

    return sensitivity_df


# =============================================================================
# STEP 5 -- PLOTS  (Updated with custom font sizes and PDF output)
# =============================================================================

def _plot_loadings(loadings, output_dir):
    """
    Bar chart: loading of each metric on each PC.
    """
    fig, ax = plt.subplots(figsize=(4.0, 3.0))  # Dimensions unchanged

    # --- ADJUST FONT SIZES HERE EASILY ---
    FS_LABEL  = 15
    FS_TICK   = 12
    FS_LEGEND = 10

    metric_labels = [METRIC_LABELS[m] for m in PCA_METRICS]
    x = np.arange(len(metric_labels))
    width = 0.27

    palette_pc = ["#868684", "#1065AB", "#B31529"]    # PC1, PC2, PC3

    for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
        ax.bar(x + (i - 1) * width, loadings[i, :], width,
               color=palette_pc[i], edgecolor='black', linewidth=0.5,
               label=pc)

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=0, fontsize=FS_TICK, fontweight="bold")
    ax.set_yticklabels(ax.get_yticks(), fontsize=FS_TICK, fontweight="bold")
    
    ax.set_ylabel('Loading on z-scored metric', fontsize=FS_LABEL, fontweight="bold")
    ax.set_ylim(-1.0, 1.0)
    ax.legend(loc='lower right', frameon=False, fontsize=FS_LEGEND)

    sns.despine(ax=ax)
    fig.tight_layout()

    path = os.path.join(output_dir, "Plot_PCA_Loadings.pdf")  # Saved as PDF
    fig.savefig(path, dpi=plotting.PLOT_STYLE['dpi_raster'], bbox_inches='tight')
    plt.close(fig)
    print(f"      -> Saved: {path}")


def _plot_scree(var_ratio, output_dir):
    """
    Scree plot: explained variance per PC, with cumulative line.
    """
    fig, ax = plt.subplots(figsize=(3.2, 2.6))  # Dimensions unchanged

    # --- ADJUST FONT SIZES HERE EASILY ---
    FS_LABEL = 15
    FS_TICK  = 12
    FS_ANNOT = 10

    pcs = ['PC1', 'PC2', 'PC3']
    pct = var_ratio * 100
    cum = np.cumsum(pct)

    ax.bar(pcs, pct, color='#42526E', edgecolor='black', linewidth=0.5,
           label='Per component')
    ax2 = ax.twinx()
    ax2.plot(pcs, cum, color='#B33A3A', marker='o', linewidth=1.0,
             markersize=4, label='Cumulative')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Cumulative (%)', fontsize=FS_LABEL, fontweight="bold")
    
    plt.setp(ax.get_xticklabels(), fontsize=FS_TICK, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=FS_TICK, fontweight="bold")
    plt.setp(ax2.get_yticklabels(), fontsize=FS_TICK, fontweight="bold")

    # Numerical labels on each bar.
    for i, v in enumerate(pct):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha='center', fontsize=FS_ANNOT, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel('Explained variance (%)', fontsize=FS_LABEL, fontweight="bold")

    sns.despine(ax=ax, right=False)
    fig.tight_layout()

    path = os.path.join(output_dir, "Plot_PCA_Scree.pdf")  # Saved as PDF
    fig.savefig(path, dpi=plotting.PLOT_STYLE['dpi_raster'], bbox_inches='tight')
    plt.close(fig)
    print(f"      -> Saved: {path}")


def _plot_scatter(scores_df, output_dir):
    """
    PC1 vs PC2 scatter with marginal KDEs, coloured by condition.
    """
    plot_df = scores_df[['Category', 'PC1', 'PC2']].copy()
    plot_df['ConditionLabel'] = plot_df['Category'].map(plotting.CONDITION_LABELS)

    palette = {
        plotting.CONDITION_LABELS[c]: plotting.CONDITION_PALETTE[c]
        for c in CORTEX_FORMING_FILTER.keys()
    }
    hue_order = [plotting.CONDITION_LABELS[c]
                 for c in ['Factin', 'BranchedCortex', 'LinearCortex']]

    # --- ADJUST FONT SIZES HERE EASILY ---
    FS_LABEL  = 15
    FS_TICK   = 15
    FS_LEGEND = 15

    g = sns.jointplot(
        data=plot_df, x='PC1', y='PC2', hue='ConditionLabel',
        hue_order=hue_order, palette=palette,
        kind='scatter', height=4.2, ratio=4, space=0.05,  # Dimensions unchanged
        marginal_kws=dict(fill=True, alpha=0.45, linewidth=0.8, common_norm=False),
        joint_kws=dict(s=8, alpha=0.5, edgecolor='none'),
    )

    g.ax_joint.axhline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.5)
    g.ax_joint.axvline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.5)

    g.ax_joint.set_xlabel('PC1  (cortex maturity)', fontsize=FS_LABEL, fontweight="bold")
    g.ax_joint.set_ylabel('PC2  (architectural identity)', fontsize=FS_LABEL, fontweight="bold")
    
    g.ax_joint.tick_params(axis='both', labelsize=FS_TICK)
    plt.setp(g.ax_joint.get_xticklabels(), fontweight="bold")
    plt.setp(g.ax_joint.get_yticklabels(), fontweight="bold")

    # --- LEGEND MOVED TO UPPER LEFT ---
    legend = g.ax_joint.legend(loc='lower left', frameon=False, fontsize=FS_LEGEND)
    
    # Robustly resize legend markers
    if legend:
        for handle in legend.legend_handles:
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([150])          # For collection handles
            elif hasattr(handle, 'set_markersize'):
                handle.set_markersize(12)        # For Line2D handles

    sns.despine(ax=g.ax_joint)

    path = os.path.join(output_dir, "Plot_PCA_Scatter.pdf")  # Saved as PDF
    g.fig.savefig(path, dpi=plotting.PLOT_STYLE['dpi_raster'], bbox_inches='tight')
    plt.close(g.fig)
    print(f"      -> Saved: {path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_pca_analysis(df, output_dir):
    """
    Runs the full PCA pipeline.
    """
    print("\n--- Running Principal Component Analysis ---")

    pca_dir = os.path.join(output_dir, "PCA_Analysis")
    os.makedirs(pca_dir, exist_ok=True)

    plotting.set_paper_style()

    required = {'Category', 'Phenotype_Category'} | set(PCA_METRICS)
    missing = required - set(df.columns)
    if missing:
        print(f"  ! Required columns missing: {sorted(missing)}")
        print("  ! Run analysis_phenotype before analysis_pca.")
        return

    subset = build_cortex_forming_subset(df)
    if len(subset) < 100:
        print(f"  ! Cortex-forming subset has only {len(subset)} vesicles.")
        print("  ! PCA aborted (too few observations for a stable fit).")
        return

    scores_df, loadings, var_ratio = fit_pca(subset, pca_dir)
    run_pca_statistical_tests(scores_df, pca_dir)
    run_bootstrap_sensitivity(subset, pca_dir)

    print("  -> Generating PCA plots...")
    _plot_loadings(loadings, pca_dir)
    _plot_scree(var_ratio, pca_dir)
    _plot_scatter(scores_df, pca_dir)

    print("  -> PCA Analysis Complete.\n")


# =============================================================================
# STANDALONE MODE
# =============================================================================

if __name__ == "__main__":
    print("Running analysis_pca.py in standalone mode...")

    if not os.path.exists(STANDALONE_CSV_PATH):
        print(f"\nERROR: Could not find CSV at:\n  {STANDALONE_CSV_PATH}")
        print("Update STANDALONE_CSV_PATH at the top of this file,")
        print("or run master_pipeline.py first to generate the phenotype CSV.\n")
    else:
        df  = pd.read_csv(STANDALONE_CSV_PATH)
        out = STANDALONE_OUTPUT_DIR or os.path.dirname(STANDALONE_CSV_PATH)
        print(f"Loaded {len(df)} vesicles.\n")
        run_pca_analysis(df, out)
        print("Done.")