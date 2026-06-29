# -*- coding: utf-8 -*-
"""
BATCH ANALYSIS MODULE
Detects and visualises batch-to-batch variability within each condition.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal, skew   # kruskal: non-parametric test for >2 groups
import plotting


# =============================================================================
# WHICH METRICS TO COMPARE ACROSS BATCHES
# =============================================================================

# These metrics are meaningful for EVERY condition (controls and experimental).
# 'Refined Radius (um)' tells you vesicle size — should be stable batch-to-batch.
# 'Solidity' tells you shape quality — also should be stable.
METRICS_ALL_CONDITIONS = [
    'Refined Radius (um)',
    'Solidity',
]

# These extra metrics only exist when actin data is present.
METRICS_ACTIN_CONDITIONS = [
    'A localization',
    'Gini_Index',
    'ISM',
    't_cortex',
]

# The set of conditions that have actin channels.
CONDITIONS_WITH_ACTIN = {'BranchedCortex', 'LinearCortex', 'Factin'}

# Human-readable axis labels for the metrics.
# (Keeps the axis tidy — shorter than the raw column name.)
METRIC_LABELS = {
    'Refined Radius (um)':  'GUV radius (µm)',
    'Solidity':             'Solidity',
    'A localization':       'Localization Score',
    'Gini_Index':           'Gini Index',
    'ISM':                  'ISM',
    't_cortex':             'Cortex Thickness (µm)',
}

# Below this many points, sample skewness is too noisy to report.
# Mirrors MIN_N_FOR_SHAPE in analysis_distribution_shape.py so the same
# "don't trust shape moments on tiny samples" rule applies everywhere.
MIN_N_FOR_SKEW = 8

# Phenotype labels that do NOT count as "a cortex formed".
# Must match the `non_cortex` set in analysis_statistics.py's CortexOnly
# summary table — both must stay in sync if phenotype labels ever change.
NON_CORTEX_PHENOTYPES = {'Empty', 'Lumenal', 'Excluded'}

# =============================================================================
# REPRESENTATIVE RADIAL PROFILE COMPARISON (Lumenal / Sparse / Continuous)
# =============================================================================

# Only these two conditions actually have a Sparse/Continuous cortex
# gradient worth comparing this way (Factin only has Shell/Lumenal — see
# analysis_phenotype.py's _classify_factin).
RADIAL_PROFILE_CONDITIONS = ['BranchedCortex', 'LinearCortex']

# The three phenotypes whose representative actin profile we want to
# compare directly. Order here is also the legend/plotting order.
RADIAL_PROFILE_PHENOTYPES = ['Lumenal', 'Sparse', 'Continuous']

# Each vesicle's radial profile is re-expressed as a FRACTION of that
# vesicle's own refined radius (radius_um / refined_radius_um), so that
# small and large vesicles can be averaged together with their membranes
# aligned at the same x-position. We then resample every vesicle onto this
# shared grid before averaging.
#
# *** Keep NORMALIZED_RADIUS_MAX in sync with 'length_excess' in main.py's
#     ANALYSIS_CONFIG (radial profiles are only sampled out to
#     length_excess times the vesicle radius, so asking for points beyond
#     that would just be flat-line extrapolation). ***
NORMALIZED_RADIUS_MAX    = 1.2
NORMALIZED_RADIUS_POINTS = 200

# Don't bother plotting a "representative" curve from fewer than this many
# vesicles — the median of 1-2 noisy profiles isn't representative of
# anything. (Sparse is the smallest cortex-forming grade in BranchedCortex,
# so this guard matters most for that group.)
MIN_N_FOR_REPRESENTATIVE_PROFILE = 3


# =============================================================================
# HELPER — DECIDE WHICH METRICS TO USE FOR A CONDITION
# =============================================================================

def _get_metrics_for_condition(condition, df_columns):
    """
    Returns the list of metric names relevant for the given condition,
    filtered to only those that actually exist as columns in the data.

    WHY FILTER AGAINST df_columns?
    If a condition (like Empty) has no actin, the actin columns simply
    won't exist — asking for them would crash the code. This function
    prevents that crash by only returning columns that are present.

    Parameters
    ----------
    condition  : str  — e.g. 'BranchedCortex'
    df_columns : list — column names that exist in the DataFrame

    Returns
    -------
    metrics : list of str
    """
    # Start with the universal metrics
    metrics = list(METRICS_ALL_CONDITIONS)

    # Add actin metrics only for conditions that have an actin channel
    if condition in CONDITIONS_WITH_ACTIN:
        metrics += METRICS_ACTIN_CONDITIONS

    # Keep only columns that actually exist — avoids KeyError crashes
    metrics = [m for m in metrics if m in df_columns]

    return metrics


# =============================================================================
# STEP 1 — BATCH STATISTICS CSV
# =============================================================================

def compute_batch_statistics(df, output_dir):
    """
    Computes mean, standard deviation, median, skewness, and N for each
    metric, grouped by (Condition, Batch_ID).

    THINK OF IT LIKE:
    A school report card where each row is one class (batch),
    each column is one subject (metric), and the value is the class average.

    WHY SKEWNESS TOO?
    ------------------
    Mean/median/std only describe the CENTRE of a batch's distribution.
    Two batches can have identical medians while one has a long tail of
    outlier vesicles and the other doesn't — the Kruskal-Wallis test in
    run_batch_significance_tests() below is mostly sensitive to shifts in
    location/rank, not to this kind of shape difference. Reporting
    per-batch skewness (same convention as Distribution_Shape_PerCondition.csv:
    scipy.stats.skew, bias-corrected) gives a second, independent way to
    spot a batch that looks "off" even when its median is unremarkable.
    Skipped (NaN) below MIN_N_FOR_SKEW points, since sample skewness is
    unreliable on small batches.

    Parameters
    ----------
    df         : pandas DataFrame  — must have 'Category' and 'Batch_ID'
    output_dir : str               — folder to save the CSV

    Returns
    -------
    stats_df : pandas DataFrame with the computed statistics
    """
    print("  -> Computing batch statistics...")

    # Combine both metric lists, then filter to those present in this dataset
    all_metrics = METRICS_ALL_CONDITIONS + METRICS_ACTIN_CONDITIONS
    all_metrics = [m for m in all_metrics if m in df.columns]

    rows = []

    # groupby() splits the DataFrame into sub-tables, one per unique
    # (Category, Batch_ID) pair, then loops through them.
    for (condition, batch_id), group in df.groupby(['Category', 'Batch_ID']):

        row = {
            'Condition': condition,
            'Batch_ID':  batch_id,
            'N_Vesicles': len(group),   # how many vesicles in this batch
        }

        for metric in all_metrics:
            if metric not in group.columns:
                continue
            values = group[metric].dropna()   # ignore missing (NaN) values

            if len(values) == 0:
                # No valid data for this metric in this batch
                row[f'{metric}_mean']   = np.nan
                row[f'{metric}_std']    = np.nan
                row[f'{metric}_median'] = np.nan
                row[f'{metric}_skew']   = np.nan
                row[f'{metric}_N']      = 0
            else:
                row[f'{metric}_mean']   = round(float(values.mean()),   4)
                row[f'{metric}_std']    = round(float(values.std()),    4)
                row[f'{metric}_median'] = round(float(values.median()), 4)

                # Skewness needs a minimum sample size to be trustworthy.
                if len(values) >= MIN_N_FOR_SKEW:
                    row[f'{metric}_skew'] = round(
                        float(skew(values.values, bias=False)), 4
                    )
                else:
                    row[f'{metric}_skew'] = np.nan

                row[f'{metric}_N']      = int(len(values))

        rows.append(row)

    stats_df = pd.DataFrame(rows).sort_values(['Condition', 'Batch_ID'])

    path = os.path.join(output_dir, "Batch_Statistics_Summary.csv")
    stats_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    return stats_df


# =============================================================================
# STEP 2 — KRUSKAL-WALLIS CONSISTENCY TESTS
# =============================================================================

def run_batch_significance_tests(df, output_dir):
    """
    Tests whether batches within each condition are statistically
    consistent using the Kruskal-Wallis test, and reports an effect size
    (epsilon-squared) alongside the p-value.

    WHAT IS THE KRUSKAL-WALLIS TEST?
    ---------------------------------
    It is a non-parametric test (does not assume your data is normally
    distributed) that asks: "Are all these groups drawn from the same
    distribution, or is at least one group different?"

    - p-value ≥ 0.05  →  batches are CONSISTENT  (no significant difference)
    - p-value < 0.05  →  batches are INCONSISTENT  (something changed)

    ANALOGY:
    You weigh 5 bags of flour from 5 different factory runs.
    The test asks: "Is one delivery systematically heavier than the others?"

    WHY ADD EPSILON-SQUARED?
    -------------------------
    At the sample sizes here (hundreds to thousands of vesicles per batch),
    the Kruskal-Wallis test is sensitive enough that p < 0.05 is the norm,
    not the exception — a significant p-value alone doesn't tell you
    whether batch identity explains 2% or 40% of the variance. Epsilon-
    squared is the standard KW effect size (Tomczak & Tomczak, 2014):

        epsilon^2 = (H - k + 1) / (n - k)

    where H is the KW statistic, k is the number of batches compared, and
    n is the total number of vesicles pooled across those batches. It is
    bounded below at 0 (negative raw values, which can occur when H is
    small relative to k, are clipped to 0) and is reported here as a
    percentage so it reads directly as "% of variance explained by batch".

    IMPORTANT CAVEAT:
    A significant result (p < 0.05) does NOT mean the experiment is ruined.
    It just means you should LOOK at the batch plot (and at epsilon-squared
    here) and decide whether the difference is scientifically meaningful.

    Parameters
    ----------
    df         : pandas DataFrame
    output_dir : str

    Returns
    -------
    list of result dicts (also saved to CSV)
    """
    print("  -> Running batch consistency tests (Kruskal-Wallis)...")

    results = []

    for condition in df['Category'].unique():
        cond_df  = df[df['Category'] == condition]
        batches  = cond_df['Batch_ID'].unique()

        # Need at least 2 batches to make a comparison
        if len(batches) < 2:
            print(f"      -> Skipping {condition}: only 1 batch found")
            continue

        metrics = _get_metrics_for_condition(condition, df.columns.tolist())

        for metric in metrics:

            # Collect the values for each batch as a separate array
            # (kruskal() needs them as separate arguments, not one big list)
            groups = []
            for batch in batches:
                vals = (cond_df
                        .loc[cond_df['Batch_ID'] == batch, metric]
                        .dropna()
                        .values)
                # Only include a batch if it has at least 3 data points
                if len(vals) >= 3:
                    groups.append(vals)

            # Can't run the test with fewer than 2 valid groups
            if len(groups) < 2:
                continue

            try:
                # The * unpacks the list: kruskal(group1, group2, group3, ...)
                stat, p_val = kruskal(*groups)

                # --- Effect size: epsilon-squared ----------------------
                k = len(groups)
                n_total = sum(len(g) for g in groups)
                if n_total > k:
                    eps_sq = (stat - k + 1) / (n_total - k)
                    eps_sq = max(eps_sq, 0.0)   # clip negative values to 0
                else:
                    eps_sq = np.nan

                # Translate p-value into a plain English interpretation
                if p_val < 0.001:
                    interp = "Highly inconsistent ***"
                elif p_val < 0.01:
                    interp = "Inconsistent **"
                elif p_val < 0.05:
                    interp = "Borderline inconsistent *"
                else:
                    interp = "Consistent (no significant difference)"

                results.append({
                    'Condition':            condition,
                    'Metric':                metric,
                    'N_Batches':             k,
                    'N_Total':               n_total,
                    'KW_statistic':          round(float(stat),  4),
                    'p_value':               round(float(p_val), 6),
                    'Epsilon_Squared':       round(float(eps_sq), 4) if not np.isnan(eps_sq) else np.nan,
                    'Epsilon_Squared_Pct':   round(float(eps_sq) * 100, 2) if not np.isnan(eps_sq) else np.nan,
                    'Interpretation':       interp,
                })

            except Exception as e:
                print(f"      ! Test failed ({condition}, {metric}): {e}")

    if results:
        results_df = (pd.DataFrame(results)
                      .sort_values(['Condition', 'p_value']))

        path = os.path.join(output_dir, "Batch_Consistency_Tests.csv")
        results_df.to_csv(path, index=False)
        print(f"      -> Saved: {path}")

        # Print a readable summary in the console
        # ✓ = consistent  |  ⚠ = inconsistent (may need attention)
        print("\n      Batch consistency summary:")
        for _, row in results_df.iterrows():
            flag = "  ⚠" if row['p_value'] < 0.05 else "  ✓"
            eps_str = (f"{row['Epsilon_Squared_Pct']:5.1f}%"
                       if pd.notna(row['Epsilon_Squared_Pct']) else "  n/a")
            print(f"{flag}  {row['Condition']:>15}  |  "
                  f"{row['Metric']:>22}  |  "
                  f"p = {row['p_value']:.4f}  |  "
                  f"eps^2 = {eps_str}  |  "
                  f"{row['Interpretation']}")
        print()

    else:
        print("      -> Not enough batches for statistical testing "
              "(need ≥ 2 batches per condition).")

    return results


# =============================================================================
# STEP 3 — PER-BATCH CORTEX-FORMING FRACTION
# =============================================================================

def compute_batch_cortex_forming_fractions(df, output_dir):
    """
    Computes the cortex-forming fraction within each (Condition, Batch_ID)
    group.

    "Cortex-forming" uses the exact same definition as the CortexOnly
    summary table in analysis_statistics.py:

        cortex-forming  =  Phenotype_Category NOT IN {'Empty', 'Lumenal',
                                                        'Excluded'}

    i.e. Sparse / Patchy / Continuous (and Shell, for F-actin) count as
    cortex-forming; Empty, Lumenal, and quality-Excluded vesicles do not.
    NON_CORTEX_PHENOTYPES at the top of this file must stay in sync with
    the `non_cortex` set in analysis_statistics.py if phenotype labels
    ever change.

    WHY THIS IS A SEPARATE QUESTION FROM THE KW TESTS ABOVE
    ----------------------------------------------------------
    run_batch_significance_tests() asks: "Among vesicles that DID form a
    cortex, are A_loc / Gini / t_cortex consistent across batches?"
    This function asks a logically separate question: "Did the SAME
    FRACTION of vesicles form a cortex at all in each batch?" A batch can
    score perfectly consistent on the first question while swinging
    wildly on the second (e.g. one prep yields very few cortex-forming
    GUVs but the ones that do form a cortex look just like every other
    batch's).

    Parameters
    ----------
    df         : pandas DataFrame — must have 'Category', 'Batch_ID',
                 'Phenotype_Category'
    output_dir : str — folder to save the CSV

    Returns
    -------
    fractions_df : pandas DataFrame, one row per (Condition, Batch_ID).
                   Empty DataFrame if 'Phenotype_Category' is missing.
    """
    print("  -> Computing per-batch cortex-forming fractions...")

    if 'Phenotype_Category' not in df.columns:
        print("      ! 'Phenotype_Category' not found — skipping "
              "(run analysis_phenotype.py before analysis_batch.py).")
        return pd.DataFrame()

    rows = []
    for (condition, batch_id), group in df.groupby(['Category', 'Batch_ID']):
        n_total  = len(group)
        n_cortex = int((~group['Phenotype_Category']
                        .isin(NON_CORTEX_PHENOTYPES)).sum())
        pct = round(100 * n_cortex / n_total, 2) if n_total > 0 else np.nan

        rows.append({
            'Condition':         condition,
            'Batch_ID':          batch_id,
            'N_Total':           n_total,
            'N_CortexForming':   n_cortex,
            'Pct_CortexForming': pct,
        })

    fractions_df = pd.DataFrame(rows).sort_values(['Condition', 'Batch_ID'])

    path = os.path.join(output_dir, "Batch_CortexForming_Fractions.csv")
    fractions_df.to_csv(path, index=False)
    print(f"      -> Saved: {path}")

    # Console summary: the min-to-max range per condition is exactly the
    # number you'd cite in text as "spanning ~X-Y% across experiments".
    print("\n      === Per-batch cortex-forming fraction range ===")
    for condition, sub in fractions_df.groupby('Condition'):
        valid = sub['Pct_CortexForming'].dropna()
        if len(valid) < 2:
            continue
        lo, hi = valid.min(), valid.max()
        print(f"      {condition:>15}  |  range: {lo:5.1f}% - {hi:5.1f}%  "
              f"(N_batches={len(valid)})")
    print()

    return fractions_df


# =============================================================================
# STEP 4 — SHORTEN BATCH LABELS FOR PLOT AXES
# =============================================================================

def _shorten_batch_label(batch_id, condition):
    """
    Converts a long batch folder name into a short axis label.

    Example:
        '260208_BranchedCortex_1'  +  'BranchedCortex'
        → '260208 #1'

    This keeps the x-axis readable even when batch names are long.

    Parameters
    ----------
    batch_id  : str — e.g. '260208_BranchedCortex_1'
    condition : str — e.g. 'BranchedCortex'

    Returns
    -------
    str — short label, e.g. '260208 #1'
    """
    # Remove the condition name from the batch ID (case-insensitive)
    short = batch_id.replace(condition, '').replace('__', '_').strip('_')

    # Split by '_' and try to find a date-like part and a run number
    parts = [p for p in short.split('_') if p]

    if len(parts) >= 2:
        # First part is typically the date, last part the run number
        return f"{parts[0]} #{parts[-1]}"
    elif len(parts) == 1:
        return parts[0]
    else:
        return batch_id   # fall back to the full name if parsing fails


def _interpolate_one_vesicle(normalized_radius, intensity, grid):
    """
    Resamples ONE vesicle's intensity curve onto the shared `grid` of
    normalized radius values, using straight-line interpolation.

    THE ANALOGY: imagine every vesicle's profile as a hand-drawn line on
    its own sheet of graph paper, where the x-axis spacing is slightly
    different on every sheet (because vesicles have different sizes).
    Before you can average many sheets together, you first need to redraw
    every line onto IDENTICAL graph paper. That's what np.interp does
    here — it reads the original (slightly irregular) line and re-draws
    it at exactly the x-positions given in `grid`.

    Parameters
    ----------
    normalized_radius : array — this vesicle's own (radius_um / refined_radius_um)
                         values; must be sorted ascending (it already is,
                         since radius_um increases outward along the profile)
    intensity          : array — the matching intensity values (membrane or actin)
    grid               : array — the shared x-axis (normalized radius) every
                         vesicle gets resampled onto

    Returns
    -------
    array, same length as `grid` — the interpolated intensity curve.
    Points in `grid` that fall outside this vesicle's own measured range
    are filled with that vesicle's first/last measured value (np.interp's
    default "clamp to the edges" behaviour) rather than wild extrapolation.
    """
    return np.interp(grid, normalized_radius, intensity)


def compute_representative_radial_profiles(df, radial_df, output_dir):
    """
    Builds a "representative" radial actin-intensity curve for the
    Lumenal / Sparse / Continuous phenotypes, separately for BranchedCortex
    and LinearCortex — the data behind the comparison plot Nikki asked for.

    THE BIG PICTURE / WHY THIS IS TRICKIER THAN A SIMPLE AVERAGE:
    Every vesicle has its OWN radial profile, sampled at its OWN set of
    radius values in micrometres (because vesicles vary in size, the
    sampled radius range differs from one vesicle to the next). You can't
    just average "intensity at radius_um = 4.0" across 30 vesicles of
    different sizes — for a small vesicle, 4.0 µm might be near the
    membrane, while for a large vesicle it might be deep in the lumen.

    The fix is to first divide each vesicle's radius_um by ITS OWN refined
    radius (from Analysis_Results.csv), turning "radius in µm" into
    "fraction of this vesicle's radius" — e.g. 1.0 always means "right at
    this vesicle's own membrane". THEN every vesicle's curve is resampled
    (interpolated) onto one shared grid of these fractions, so they can
    finally be stacked together and averaged point-by-point.

    Parameters
    ----------
    df         : pandas DataFrame — master dataset, already passed through
                 analysis_phenotype.run_phenotype_analysis() so that it has
                 'Phenotype_Category' (and 'Refined Radius (um)').
    radial_df  : pandas DataFrame, or None — output of
                 file_handling.load_radial_profiles(). If None, this whole
                 step is skipped (with an explanatory message) so the rest
                 of the pipeline still runs.
    output_dir : str — folder to save Representative_Radial_Profiles.csv into

    Returns
    -------
    rep_df : pandas DataFrame in "tidy" long format, with one row per
             (Condition, Phenotype, Normalized_Radius) grid point:
                 Condition, Phenotype, N_vesicles, Normalized_Radius,
                 Median_Actin, Q1_Actin, Q3_Actin,
                 Median_Membrane, Q1_Membrane, Q3_Membrane
             or None if radial_df was None or nothing could be matched.
    """
    print("\n--- Computing Representative Radial Profiles ---")

    if radial_df is None:
        print("  ! No radial profile data available "
              "(file_handling.load_radial_profiles returned None). Skipping.")
        return None

    required_cols = {'Category', 'Batch_ID', 'Region_ID', 'Vesicle id',
                      'Phenotype_Category', 'Refined Radius (um)'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  ! Master dataset is missing column(s) {missing}. "
              "Make sure phenotype analysis has already run. Skipping.")
        return None

    # ── Step 1: join the radius-point-level data with each vesicle's
    #            phenotype label and refined radius ──────────────────────
    # We match rows on the SAME 4 columns that uniquely identify a vesicle
    # everywhere else in this pipeline. 'Vesicle id' is cast to a plain
    # int on both sides first, since one side may have come back from a
    # CSV as a float (e.g. 3.0) and the other as an int (e.g. 3) — those
    # would otherwise fail to match during the merge.
    id_cols = ['Category', 'Batch_ID', 'Region_ID', 'Vesicle id']

    left = radial_df.copy()
    left['Vesicle id'] = pd.to_numeric(left['Vesicle id'], errors='coerce').round().astype('Int64')

    right = df[id_cols + ['Phenotype_Category', 'Refined Radius (um)']].copy()
    right['Vesicle id'] = pd.to_numeric(right['Vesicle id'], errors='coerce').round().astype('Int64')

    merged = left.merge(right, on=id_cols, how='inner')

    if merged.empty:
        print("  ! Merge between radial profiles and the master dataset "
              "produced 0 matching rows — check that both were generated "
              "from the same Output folder. Skipping.")
        return None

    # ── Step 2: keep only rows we can actually normalise ──────────────────
    merged = merged[merged['Refined Radius (um)'] > 0].copy()
    merged['Normalized_Radius'] = merged['radius_um'] / merged['Refined Radius (um)']

    # The shared x-axis grid every vesicle's curve gets resampled onto.
    grid = np.linspace(0, NORMALIZED_RADIUS_MAX, NORMALIZED_RADIUS_POINTS)

    rep_rows = []

    for condition in RADIAL_PROFILE_CONDITIONS:
        cond_df = merged[merged['Category'] == condition]

        for phenotype in RADIAL_PROFILE_PHENOTYPES:
            group = cond_df[cond_df['Phenotype_Category'] == phenotype]

            # Each vesicle contributes ONE interpolated curve. group.groupby
            # walks through the data vesicle-by-vesicle (grouped by the same
            # 4 ID columns), so 'vesicle' below is one vesicle's full set
            # of radius/intensity points, already sorted by radius_um.
            actin_curves    = []
            membrane_curves = []
            for _, vesicle in group.groupby(id_cols):
                vesicle = vesicle.sort_values('Normalized_Radius')
                actin_curves.append(_interpolate_one_vesicle(
                    vesicle['Normalized_Radius'].values,
                    vesicle['Actin_Intensity'].values, grid))
                membrane_curves.append(_interpolate_one_vesicle(
                    vesicle['Normalized_Radius'].values,
                    vesicle['Membrane_Intensity'].values, grid))

            n_vesicles = len(actin_curves)
            if n_vesicles < MIN_N_FOR_REPRESENTATIVE_PROFILE:
                print(f"  -> Skipping {condition} / {phenotype}: "
                      f"only {n_vesicles} vesicle(s) "
                      f"(need >= {MIN_N_FOR_REPRESENTATIVE_PROFILE}).")
                continue

            # Stack every vesicle's curve into one 2D array
            # (n_vesicles rows x NORMALIZED_RADIUS_POINTS columns), then
            # take the median/IQR DOWN the rows (axis=0) — i.e. across
            # vesicles, separately at every grid point.
            actin_stack    = np.vstack(actin_curves)
            membrane_stack = np.vstack(membrane_curves)

            for i, x in enumerate(grid):
                rep_rows.append({
                    'Condition':         condition,
                    'Phenotype':         phenotype,
                    'N_vesicles':        n_vesicles,
                    'Normalized_Radius': round(float(x), 4),
                    'Median_Actin':      float(np.median(actin_stack[:, i])),
                    'Q1_Actin':          float(np.percentile(actin_stack[:, i], 25)),
                    'Q3_Actin':          float(np.percentile(actin_stack[:, i], 75)),
                    'Median_Membrane':   float(np.median(membrane_stack[:, i])),
                    'Q1_Membrane':       float(np.percentile(membrane_stack[:, i], 25)),
                    'Q3_Membrane':       float(np.percentile(membrane_stack[:, i], 75)),
                })

            print(f"  -> {condition:>14} / {phenotype:<10}: "
                  f"{n_vesicles} vesicles")

    if not rep_rows:
        print("  ! No phenotype group had enough vesicles to build a "
              "representative profile. Skipping plot.")
        return None

    rep_df = pd.DataFrame(rep_rows)

    # ── Save the underlying numbers, so every curve in the final figure
    #    can be traced back to a CSV row (same "verify at source" workflow
    #    used everywhere else in this pipeline) ───────────────────────────
    rep_csv_path = os.path.join(output_dir, "Representative_Radial_Profiles.csv")
    rep_df.to_csv(rep_csv_path, index=False)
    print(f"  -> Saved: {rep_csv_path}")

    # ── Draw the comparison figure ─────────────────────────────────────────
    plotting.plot_representative_radial_profiles(rep_df, output_dir)

    print("  -> Representative Radial Profile comparison complete.")
    return rep_df


# =============================================================================
# REPRESENTATIVE CHANNEL IMAGES (Empty / Lumenal / Sparse / Patchy / Continuous)
# =============================================================================

# Real biological phenotype labels worth showing a picture of. 'Excluded'
# is deliberately left out — it's a QC catch-all (too small, no membrane
# detected, wide-peak artefact, ...) covering several unrelated failure
# modes, not a single recognisable "look", so a single representative
# image of it wouldn't mean much.
REPRESENTATIVE_IMAGE_PHENOTYPES_TO_SKIP = {'Excluded'}

# Tried in this order to pick "the" representative vesicle within a
# (Condition, Phenotype) group: the first metric that's actually present
# AND has more than one distinct value in this group (so we don't try to
# rank vesicles by a column that's all zeros, e.g. localization for a
# group that's all Empty). Falls back to "just the lowest Vesicle id" if
# none of these work, so the function always returns a pick rather than
# raising an error.
_SELECTION_METRIC_PRIORITY = ['A localization', 'Gini_Index', 'Refined Radius (um)']


def select_representative_vesicles(df, output_dir):
    """
    Picks ONE "typical" vesicle for every (Category, Phenotype_Category)
    combination actually present in the data, and saves the picks to a CSV
    so each one can be inspected/verified by hand later.

    THE ANALOGY: imagine picking one photo to represent "what a B+ student
    looks like" out of a whole class's photos — you wouldn't pick the best
    or the worst, you'd pick whoever is closest to the B+ average. That's
    exactly what this does: for each group, it ranks vesicles by whichever
    metric defines that phenotype most directly (A localization first,
    since that's literally the axis analysis_phenotype.py's classifiers
    use), and picks the one closest to the GROUP'S OWN MEDIAN.

    This function only SELECTS — it doesn't touch any image files. The
    actual crop images are loaded later, by
    plotting.plot_representative_channel_images(), using the Batch_ID /
    Region_ID / Vesicle id columns saved here to rebuild each file's path.

    Parameters
    ----------
    df         : pandas DataFrame — master dataset, already classified by
                 analysis_phenotype.run_phenotype_analysis() (must have
                 'Phenotype_Category').
    output_dir : str — folder to save Representative_Vesicles.csv into

    Returns
    -------
    rep_vesicles_df : pandas DataFrame with one row per (Category,
                      Phenotype_Category) combination:
                          Category, Phenotype_Category, Batch_ID,
                          Region_ID, Vesicle id, N_in_group, Selection_Metric
                      or None if 'Phenotype_Category' isn't in df.
    """
    print("\n--- Selecting Representative Vesicles (for channel images) ---")

    if 'Phenotype_Category' not in df.columns:
        print("  ! 'Phenotype_Category' not found — run phenotype analysis first. Skipping.")
        return None

    candidates = df[~df['Phenotype_Category'].isin(REPRESENTATIVE_IMAGE_PHENOTYPES_TO_SKIP)]

    picks = []
    for (condition, phenotype), group in candidates.groupby(['Category', 'Phenotype_Category']):

        chosen_metric = None
        for metric in _SELECTION_METRIC_PRIORITY:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0 and values.nunique() > 1:
                    chosen_metric = metric
                    break

        if chosen_metric is not None:
            # The vesicle whose value sits closest to this group's own
            # median for that metric — i.e. the most "typical" member.
            median_val  = group[chosen_metric].median()
            chosen_idx  = (group[chosen_metric] - median_val).abs().idxmin()
        else:
            # No usable metric (e.g. a single-vesicle group, or every
            # value identical) — just take the lowest Vesicle id so the
            # pick is at least deterministic and reproducible.
            chosen_idx = group.sort_values('Vesicle id').index[0]

        chosen = df.loc[chosen_idx]
        picks.append({
            'Category':            condition,
            'Phenotype_Category':  phenotype,
            'Batch_ID':            chosen['Batch_ID'],
            'Region_ID':           chosen['Region_ID'],
            'Vesicle id':          int(chosen['Vesicle id']),
            'N_in_group':          len(group),
            'Selection_Metric':    chosen_metric if chosen_metric else 'lowest_vesicle_id',
        })
        print(f"  -> {condition:>14} / {phenotype:<10}: "
              f"Vesicle {int(chosen['Vesicle id'])} from {chosen['Batch_ID']} "
              f"(N={len(group)}, picked by {chosen_metric or 'lowest_vesicle_id'})")

    if not picks:
        print("  ! No groups found to pick representatives from. Skipping.")
        return None

    rep_vesicles_df = pd.DataFrame(picks)

    rep_path = os.path.join(output_dir, "Representative_Vesicles.csv")
    rep_vesicles_df.to_csv(rep_path, index=False)
    print(f"  -> Saved: {rep_path}")

    return rep_vesicles_df


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_batch_analysis(df, output_dir, radial_df=None, root_path=None):
    """
    Main function that runs ALL batch analyses in sequence.

    Call this from master_pipeline.py after the other analysis modules.

    Steps:
      1. Compute and save per-batch statistics CSV (mean/std/median/skew/N)
      2. Run Kruskal-Wallis consistency tests + epsilon-squared, save CSV
      3. Compute per-batch cortex-forming fraction, save CSV
      4. Generate one per-condition batch variability plot
      5. Generate one overview plot showing all conditions
      6. (NEW) Build the representative Lumenal/Sparse/Continuous radial
         actin-profile comparison for BranchedCortex vs LinearCortex
      7. (NEW) Pick + assemble representative membrane/actin channel
         images for every Condition x Phenotype combination

    Parameters
    ----------
    df         : pandas DataFrame — the master dataset from file_handling.py
    output_dir : str              — the Batch_Analysis_Results folder path
    radial_df  : pandas DataFrame, or None — output of
                 file_handling.load_radial_profiles(). Optional so that
                 existing calls to run_batch_analysis(df, output_dir)
                 elsewhere keep working unchanged; pass it in to also get
                 the representative radial profile comparison (Step 6).
    root_path  : str, or None — the same ROOT_PATH used everywhere else
                 (the top-level Output folder). Needed to find each
                 representative vesicle's saved crop images on disk for
                 Step 7; if None, Step 7 is skipped.
    """
    print("\n--- Running Batch-to-Batch Variability Analysis ---")

    # Safety check — tell the user clearly if something is missing
    if 'Batch_ID' not in df.columns:
        print("  ! 'Batch_ID' column not found.")
        print("    Make sure you are using the current version of file_handling.py.")
        return

    if 'Category' not in df.columns:
        print("  ! 'Category' column not found.")
        return

    # Create a dedicated sub-folder so batch outputs don't clutter the main folder
    batch_output_dir = os.path.join(output_dir, "Batch_Analysis")
    os.makedirs(batch_output_dir, exist_ok=True)

    # ── Step 1: Statistics CSV ───────────────────────────────────────────────
    compute_batch_statistics(df, batch_output_dir)

    # ── Step 2: Significance tests ───────────────────────────────────────────
    run_batch_significance_tests(df, batch_output_dir)

    # ── Step 3: Cortex-forming fraction per batch ────────────────────────────
    # Requires 'Phenotype_Category' — master_pipeline.py already runs
    # analysis_phenotype.py before analysis_batch.py, so this is normally
    # present; the function itself guards against it being absent.
    compute_batch_cortex_forming_fractions(df, batch_output_dir)

    # ── Add short batch labels to the full dataset ───────────────────────────
    # These labels (e.g. '260208 #1') are used on the x-axes of every plot.
    # We do this once here and pass the labelled DataFrame into every plot
    # function, so each function doesn't have to repeat the logic.
    plotting.set_paper_style()

    df_labelled = df.copy()
    df_labelled['Batch_Label'] = df_labelled.apply(
        lambda row: _shorten_batch_label(row['Batch_ID'], row['Category']),
        axis=1
    )

    # ── Step 4: Size distribution — all conditions, one figure ───────────────
    # Shows how vesicle RADIUS varies across batches within each condition.
    # A good experiment has boxes at similar heights in every batch.
    plotting.plot_batch_size_distribution(df_labelled, batch_output_dir)

    # ── Step 5: Phenotype composition — all conditions, one figure ────────────
    # Shows the PERCENTAGE of each phenotype per batch.
    # Requires that analysis_phenotype.run_phenotype_analysis() was called
    # before run_batch_analysis() — master_pipeline.py ensures this ordering.
    if 'Phenotype_Category' in df_labelled.columns:
        plotting.plot_batch_phenotype_composition(df_labelled, batch_output_dir)
    else:
        print("  -> Skipping phenotype composition plot "
              "('Phenotype_Category' column not found — "
              "check that phenotype analysis ran before batch analysis).")

    # ── Step 6: Actin metrics — t_cortex, localization, ISM, Gini ────────────
    # Only plotted for conditions that have an actin channel.
    # Empty is skipped automatically inside the function.
    plotting.plot_batch_actin_metrics(df_labelled, batch_output_dir)

    # ── Step 7: Cortex-forming-only companion ─────────────────────────────────
    # Same as Step 6 but EMPTY + LUMENAL GUVs are excluded first, so the
    # dashed median line and violin shapes reflect only GUVs that actually
    # formed a membrane-associated structure.
    #
    # This call lives here (not in analysis_statistics.py) because it needs
    # 'Batch_Label', which is created above in this function.
    if 'Phenotype_Category' in df_labelled.columns:
        plotting.plot_batch_actin_metrics_cortex_only(df_labelled, batch_output_dir)
    else:
        print("  -> Skipping cortex-only batch actin plot "
              "('Phenotype_Category' not found — run phenotype analysis first).")

    # ── Step 8: Representative radial profile comparison ─────────────────────
    # Lumenal vs Sparse vs Continuous, BranchedCortex vs LinearCortex,
    # built from the raw radial profiles (Radial_Intensity_Profiles.csv),
    # not from the single-number summary metrics used in Steps 1-7.
    # Requires 'Phenotype_Category' (same guard as Steps 5/7) AND radial_df.
    if radial_df is None:
        print("  -> Skipping representative radial profile comparison "
              "(no radial_df passed in — see file_handling.load_radial_profiles).")
    elif 'Phenotype_Category' not in df.columns:
        print("  -> Skipping representative radial profile comparison "
              "('Phenotype_Category' not found — run phenotype analysis first).")
    else:
        compute_representative_radial_profiles(df, radial_df, batch_output_dir)

    # ── Step 9: Representative channel images ─────────────────────────────────
    # One membrane/actin crop pair per (Condition, Phenotype) combination —
    # e.g. "what does a typical Continuous BranchedCortex GUV's actin
    # channel actually look like?" Requires 'Phenotype_Category' AND
    # root_path (to find each picked vesicle's saved crop PNG on disk).
    if 'Phenotype_Category' not in df.columns:
        print("  -> Skipping representative channel images "
              "('Phenotype_Category' not found — run phenotype analysis first).")
    elif root_path is None:
        print("  -> Skipping representative channel images "
              "(no root_path passed in — needed to locate saved crop images).")
    else:
        rep_vesicles_df = select_representative_vesicles(df, batch_output_dir)
        if rep_vesicles_df is not None:
            plotting.plot_representative_channel_images(
                rep_vesicles_df, root_path, batch_output_dir)

    print("  -> Batch Analysis Complete.\n")

# NOTE: plot_batch_actin_metrics is called from run_batch_analysis above.
# The call is appended below by patching the file end.