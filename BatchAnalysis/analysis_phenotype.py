# -*- coding: utf-8 -*-
"""
PHENOTYPE ANALYSIS MODULE
Handles phenotype classification and visualization for all 4 conditions.

CHANGES FROM PREVIOUS VERSION:
-------------------------------
1. Classification is CONDITION-AWARE.
   Each condition has its own classification function with its own criteria.

2. LinearCortex uses SPARSE / CONTINUOUS only — PATCHY has been removed.
   Gini_Index shows no discriminatory power for linear actin networks;
   all vesicles cluster at Gini 0.40–0.48 regardless of localization.

3. Factin uses SHELL and LUMENAL only.

4. Empty vesicles are labelled EMPTY (no actin analysis is meaningful).

5. NEW: All four classifiers now check 'Shape_Quality_Flag' as their first
   step. Vesicles whose refined radius is NaN or below the threshold set in
   file_handling.py are labelled EXCLUDED in every condition — including
   Empty, which previously had no EXCLUDED branch at all. This makes the
   per-condition EXCLUDED counts directly comparable and gives an honest
   accounting of how many vesicles were sub-resolution in each preparation.
   The fall-through default of True keeps the classifiers robust against
   re-runs on older CSVs that lack the flag column.
"""

import pandas as pd
import numpy as np
import os
import plotting


# =============================================================================
# 1. CLASSIFICATION FUNCTIONS — one per condition
# =============================================================================

def _classify_branched_cortex(row):
    """
    Classifies a single BranchedCortex vesicle row.

    Decision tree:
    ─ Shape_Quality_Flag == False? → EXCLUDED  (sub-threshold size; not analysable)
    ─ Is t_cortex NaN?             → EXCLUDED  (actin analysis failed for this vesicle)
    ─ Is localization <= 0?
        ─ Is lumen/bg <= 2?        → EMPTY     (no actin anywhere)
        ─ Otherwise                → LUMENAL   (actin is inside, not on membrane)
    ─ Is localization <= 0.65?     → SPARSE    (some cortex but not dense)
    ─ Is gini > 0.45?              → PATCHY    (dense but uneven / clustered)
    ─ Otherwise                    → CONTINUOUS (dense and uniform cortex)
    """
    # Size-based exclusion comes first. .get() default True means the
    # classifier still works on CSVs that pre-date the flag column.
    if not row.get('Shape_Quality_Flag', True):
        return "EXCLUDED"

    if pd.isna(row.get('t_cortex')):
        return "EXCLUDED"

    loc         = row.get('A localization', 0)
    lumen_ratio = row.get('A Lumen/Bg', 0)
    gini        = row.get('Gini_Index', 0)

    if loc <= 0.0:
        return "EMPTY" if lumen_ratio <= 2.0 else "LUMENAL"
    if loc <= 0.65:          # <-- TUNE THIS if needed for BranchedCortex
        return "SPARSE"
    return "PATCHY" if gini > 0.45 else "CONTINUOUS"


def _classify_linear_cortex(row):
    """
    Classifies a single LinearCortex vesicle row.

    PHENOTYPES (3 only — PATCHY is intentionally absent):
    ─ Shape_Quality_Flag == False? → EXCLUDED  (sub-threshold size; not analysable)
    ─ Is t_cortex NaN?             → EXCLUDED  (actin analysis failed)
    ─ Is localization <= 0?
        ─ Is lumen/bg <= 2?        → EMPTY     (no actin anywhere)
        ─ Otherwise                → LUMENAL   (actin inside, not at membrane)
    ─ Is localization <= 0.65?     → SPARSE    (partial cortex)
    ─ Otherwise                    → CONTINUOUS (well-formed linear cortex)

    WHY NO PATCHY:
    Linear actin networks (formins, fascin bundles) do not form the discrete
    nucleation patches that branched Arp2/3 networks do. Inspection of the
    Gini_Index pair plot shows all LinearCortex vesicles cluster at Gini
    0.40–0.48 regardless of localization — there is no separation between
    any sub-groups, so PATCHY carries no biological meaning here.

    The localization threshold of 0.65 reflects the natural valley between
    the two peaks visible in the A localization KDE (~0.50 and ~0.85).
    """
    if not row.get('Shape_Quality_Flag', True):
        return "EXCLUDED"

    if pd.isna(row.get('t_cortex')):
        return "EXCLUDED"

    loc         = row.get('A localization', 0)
    lumen_ratio = row.get('A Lumen/Bg', 0)

    if loc <= 0.0:
        return "EMPTY" if lumen_ratio <= 2.0 else "LUMENAL"
    if loc <= 0.65:      # <-- TUNE THIS if your data suggests a different valley
        return "SPARSE"
    return "CONTINUOUS"


def _classify_factin(row):
    """
    Classifies a single Factin vesicle into two phenotypes only:

    SHELL    — Actin is concentrated at the membrane (forms a cortex/shell).
               Indicated by a high localization score.

    LUMENAL  — Actin is distributed throughout the vesicle interior.
               Indicated by a low localization score but detectable lumen signal.

    Decision tree:
    ─ Shape_Quality_Flag == False? → EXCLUDED  (sub-threshold size; not analysable)
    ─ Is t_cortex NaN?             → EXCLUDED  (actin analysis failed)
    ─ Is localization NaN?         → EXCLUDED  (localization not computed)
    ─ Is localization > 0.30?      → SHELL
    ─ Otherwise                    → LUMENAL

    *** TUNE THE THRESHOLD BELOW ***
    A localization score > 0.30 is used as the default cut-off for SHELL.
    This means: "at least 30% more actin at the membrane than in the lumen."
    Inspect your data and adjust if needed.
    """
    if not row.get('Shape_Quality_Flag', True):
        return "EXCLUDED"

    if pd.isna(row.get('t_cortex')):
        return "EXCLUDED"

    loc = row.get('A localization', 0)

    if pd.isna(loc):
        return "EXCLUDED"

    if loc > 0.30:       # <-- TUNE THIS: threshold for 'forms a shell'
        return "SHELL"
    else:
        return "LUMENAL"


def _classify_empty(row):
    """
    Empty vesicles contain no actin, so the only meaningful distinction is
    whether they passed the size analysability threshold.

    Decision tree:
    ─ Shape_Quality_Flag == False? → EXCLUDED  (sub-threshold size; not analysable)
    ─ Otherwise                    → EMPTY

    Note: the EXCLUDED branch is necessary here even though Empty has no
    actin channel, because we want the per-condition EXCLUDED counts to
    reflect size failures consistently across all four conditions.
    """
    if not row.get('Shape_Quality_Flag', True):
        return "EXCLUDED"

    return "EMPTY"


# =============================================================================
# 2. DISPATCH — applies the right classifier based on the condition
# =============================================================================

# This dictionary maps a condition name to its classifier function.
# Adding a new condition in the future is as simple as adding one line here.
_CLASSIFIERS = {
    'BranchedCortex': _classify_branched_cortex,
    'LinearCortex':   _classify_linear_cortex,
    'Factin':         _classify_factin,
    'Empty':          _classify_empty,
}


def categorize_vesicles(df):
    """
    Applies the correct classifier to each row based on its 'Category' column.
    Adds a new column 'Phenotype_Category' to the DataFrame.

    Works row by row — for each vesicle, it looks up the right classifier
    function and calls it with that row's data.

    Parameters
    ----------
    df : pandas DataFrame — must have a 'Category' column

    Returns
    -------
    df : the same DataFrame with a new 'Phenotype_Category' column added
    """
    def dispatch(row):
        classifier = _CLASSIFIERS.get(row['Category'])
        if classifier is None:
            # Unknown condition — label as EXCLUDED
            return "EXCLUDED"
        return classifier(row)

    df['Phenotype_Category'] = df.apply(dispatch, axis=1)
    return df


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

def run_phenotype_analysis(df, output_dir):
    """
    Runs the full phenotype analysis pipeline:
      1. Classifies every vesicle
      2. Reports the EXCLUDED count per condition (size + actin failures combined)
      3. Saves a CSV with the classifications
      4. Generates all phenotype plots
    """
    print("\n--- Running Phenotype Analysis (4 Conditions) ---")

    # 0. Set the plot style
    plotting.set_paper_style()

    # 1. Classify every vesicle
    categorize_vesicles(df)

    # 1a. Report EXCLUDED counts per condition (size + actin failures combined)
    if 'Shape_Quality_Flag' in df.columns:
        print("\n  EXCLUDED breakdown per condition (size + actin failures):")
        for cat in plotting.CONDITION_ORDER:
            cond = df[df['Category'] == cat]
            if cond.empty:
                continue
            n_total      = len(cond)
            n_size_fail  = (~cond['Shape_Quality_Flag']).sum()
            n_excluded   = (cond['Phenotype_Category'] == 'EXCLUDED').sum()
            # Vesicles labelled EXCLUDED but with a valid size flag are actin
            # analysis failures (only happens in actin-containing conditions).
            n_actin_fail = n_excluded - n_size_fail
            pct = 100 * n_excluded / n_total if n_total > 0 else 0
            print(f"      {cat:>15} : "
                  f"size_fail={n_size_fail:>4}  "
                  f"actin_fail={n_actin_fail:>4}  "
                  f"total_excluded={n_excluded:>4} / {n_total} ({pct:.1f}%)")

    # 1b. Save detailed assignments to CSV
    detailed_path = os.path.join(output_dir, "Vesicle_Phenotype_Assignments.csv")
    df.to_csv(detailed_path, index=False)
    print(f"\n  -> Phenotype assignments saved: {detailed_path}")

    # Print a quick summary to the terminal
    print("\n  Phenotype counts per condition:")
    summary = df.groupby(['Category', 'Phenotype_Category']).size().unstack(fill_value=0)
    print(summary.to_string())
    print()

    # 2. Stacked bar chart showing phenotype composition per condition
    plotting.plot_phenotype_composition(
        df, 'Category', 'Phenotype_Category', output_dir,
        'Phenotype_Composition_Stacked.png'
    )

    # 3. Metric comparison matrix (box + strip + stats)
    #    Only for conditions that have multi-phenotype data
    conditions_with_phenotypes = ['BranchedCortex', 'LinearCortex', 'Factin']
    df_phenotyped = df[df['Category'].isin(conditions_with_phenotypes)]

    if not df_phenotyped.empty:
        plotting.generate_category_panel(df_phenotyped, output_dir)

    # 4. Per-condition corrplot pair plots
    #    (only where we have meaningful phenotype variation)
    for condition in conditions_with_phenotypes:
        cond_df = df[df['Category'] == condition]
        if cond_df.empty:
            continue

        # Combined corrplot: scatter (lower) + KDE (diagonal) + r= squares (upper)
        plotting.plot_pairplot(cond_df, output_dir, filename_suffix=f"_{condition}")

    print("  -> Phenotype Analysis Complete.")