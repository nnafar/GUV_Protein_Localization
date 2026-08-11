# -*- coding: utf-8 -*-
"""
PLOTTING MODULE
Handles all data visualization, standardizing output styles, layouts, 
and statistical formatting across the pipeline.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import mannwhitneyu

# ─────────────────────────────────────────────────────────────────────────────
# 0. GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

CONDITION_ORDER = ["Empty", "Factin", "BranchedCortex", "LinearCortex"]

CONDITION_LABELS = {
    "Empty":          "Empty",
    "Factin":         "F-actin",
    "BranchedCortex": "Branched",
    "LinearCortex":   "Linear",
}

# ── Shared metric display names, used by every plotting function ──────────
# Centralizing this in ONE dictionary (instead of the 8 separate copies that
# used to exist scattered through this file) means renaming a metric is a
# one-line change here, not a find-and-replace across the whole module.
#
# Subscripts use matplotlib's mathtext syntax: wrapping a fragment in $...$
# triggers math rendering, and an underscore inside that block (e.g. "t_{cortex}")
# renders as a subscript. Units like "(µm)" are kept OUTSIDE the $...$ block —
# mathtext does not reliably handle the µ symbol or spaces inside math mode.
METRIC_LABELS = {
    "t_cortex":            "t$_{cortex}$ (µm)",
    "A localization":      "A$_{loc}$",
    "Gini_Index":          "Gini",
    "A Lumen/Bg":          "A$_{Lumen/Bg}$",
    "Refined Radius (um)": "GUV Radius (µm)",
}

# PHENOTYPE_PALETTE = {
#     "Empty":          "#D9E4E9",   
#     "Excluded":       "#e8e8e8",   
#     "Lumenal":        "#AAB8C2",   
#     "Sparse":         "#6B7B8C",   
#     "Shell":          "#6B7B8C",   
#     "Patchy":         "#42526E",   
#     "Continuous":     "#233040",   
# }

PHENOTYPE_PALETTE = {
    "Excluded":       "#DBD9D4", 
    "Empty":          "#8EC4DE",     
    "Lumenal":        "#3A93C3",   
    "Shell":          "#1065AB",   
    "Sparse":         "#F6A482",   
    "Patchy":         "#D75F4C",   
    "Continuous":     "#B31529",   
}

# CONDITION_PALETTE = {
#     "Empty":          "#D9E4E9",   
#     "Factin":         "#AAB8C2",   
#     "BranchedCortex": "#42526E",   
#     "LinearCortex":   "#233040",   
# }

CONDITION_PALETTE = {
    "Empty":          "#DBD9D4",   
    "Factin":         "#868684",   
    "BranchedCortex": "#1065AB",   
    "LinearCortex":   "#B31529",   
}

# Used by plot_representative_radial_profiles(): when several conditions are
# overlaid in ONE panel, color is reserved for Phenotype (so the cortex-
# density gradient reads consistently with every other figure), and
# linestyle is what tells BranchedCortex apart from LinearCortex instead.
CONDITION_LINESTYLE = {
    "Empty":          ":",
    "Factin":         "-.",
    "BranchedCortex": "-",
    "LinearCortex":   "--",
}

# The single shared "settings sheet" for every figure.
PLOT_STYLE = {
    "font_family":          "Arial",
    "sns_style":            "ticks",
    "sns_context":          "paper",

    # ── The actual physical width your figures print at (inches) ───────────
    # Derived from: A5 page 17 cm wide x hscale 0.75 = 12.75 cm = 5.02 in.
    "page_textwidth_in":    5.02,

    # ── Target PRINTED font sizes, in points, on the final page ────────────
    "fontsize_title_pt":    12.0,
    "fontsize_label_pt":    12.0,
    "fontsize_tick_pt":     12.0,
    "fontsize_legend_pt":   12.0,
    "fontsize_annot_pt":    12.0,

    # ── Line widths (these do NOT need pt-conversion — they look the same
    #    relative to the figure regardless of canvas size) ──────────────────
    "axes_linewidth":       1.5,   # Thicker lines for visibility at small scale
    "tick_linewidth":       1.5,

    # ── Standard widths (inches) used when building each figure's canvas ───
    # A "single column" figure is one Axes (one panel) on its own.
    # Grid figures (multiple rows/cols of panels) use per_panel_width_in and
    # per_panel_height_in, multiplied by however many rows/cols that figure
    # needs — see _get_figsize_grid() below.
    "single_col_width_in":      7.0,
    "single_col_height_in":     5.5,
    "per_panel_width_in":       3.0,
    "per_panel_height_in":      3.6,

    "dpi_raster":               600,   # Passed to savefig
}


def _fs(role):
    """
    Looks up the printed point size for a given role (e.g. "label", "tick")
    and converts it into the matplotlib fontsize that will achieve that
    printed size for a figure drawn at PLOT_STYLE["single_col_width_in"].

    THE ANALOGY: think of this like sizing text for a photocopier. You tell
    this function "I want 9pt text once this is printed," and it tells
    matplotlib how big to draw the letters on the original page so that,
    after the photocopier shrinks everything down to fit, the text comes
    out at exactly 9pt.

    Parameters
    ----------
    role : str
        One of "title", "label", "tick", "legend", "annot".

    Returns
    -------
    float : the matplotlib fontsize to pass into fontsize=... calls.

    NOTE: this assumes the figure is being drawn at the standard single
    column width. For figures drawn at a custom/dynamic width (grids whose
    width depends on the number of panels), use _fs_for_width() instead so
    the conversion uses that figure's *actual* width.
    """
    return _fs_for_width(role, PLOT_STYLE["single_col_width_in"])


def _fs_for_width(role, drawn_width_in):
    """
    Same idea as _fs(), but for figures whose canvas width is computed
    dynamically (e.g. grids that grow wider with more columns). Pass in the
    actual width (in inches) the figure is being drawn at, and this returns
    the matplotlib fontsize needed to print at the target point size on the
    real page.

    Parameters
    ----------
    role : str
        One of "title", "label", "tick", "legend", "annot".
    drawn_width_in : float
        The width (inches) passed into plt.subplots(figsize=(drawn_width_in, ...))
        for this specific figure.
    """
    target_pt = PLOT_STYLE[f"fontsize_{role}_pt"]
    scale_factor = PLOT_STYLE["page_textwidth_in"] / drawn_width_in
    return target_pt / scale_factor


def _get_figsize_single_col():
    """
    Returns the standard (width, height) in inches for any single-panel
    figure (one Axes, not a grid). Centralizing this means every "normal"
    plot in the pipeline has the same canvas proportions, so fonts computed
    by _fs() are correct for ALL of them — and changing the standard size
    only requires editing PLOT_STYLE, not every plotting function.
    """
    return (PLOT_STYLE["single_col_width_in"], PLOT_STYLE["single_col_height_in"])


def _get_figsize_grid(nrows, ncols):
    """
    Returns the (width, height) in inches for a grid figure with the given
    number of rows and columns, using the per-panel size in PLOT_STYLE.

    Analogy: think of each panel as one tile, and this function just figures
    out how big the whole tiled wall needs to be once you know how many
    rows and columns of tiles you're laying down.

    Because the WIDTH changes with ncols, the fontsize needed to hit the
    same printed point size also changes — use _fs_for_width(role, width)
    with the width returned here, not the plain _fs().
    """
    width  = PLOT_STYLE["per_panel_width_in"]  * ncols
    height = PLOT_STYLE["per_panel_height_in"] * nrows
    return (width, height)


def set_paper_style():
    """
    Applies the global seaborn theme and matplotlib rcParams shared by every
    figure in the pipeline. Tick label sizes and axis title sizes are pulled
    from PLOT_STYLE via _fs(), so they automatically match the printed-size
    targets defined above for the standard single-column figure width.

    NOTE: rcParams set here apply globally as defaults. Grid figures with a
    non-standard width still call _fs_for_width(...) explicitly wherever
    they set tick/label/title sizes, which overrides these defaults with the
    correctly-scaled value for that figure's actual canvas width.
    """
    sns.set_theme(
        style   = PLOT_STYLE["sns_style"],
        context = PLOT_STYLE["sns_context"],
    )
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    [PLOT_STYLE["font_family"], "DejaVu Sans"],
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     PLOT_STYLE["axes_linewidth"],
        "xtick.major.width":  PLOT_STYLE["tick_linewidth"],
        "ytick.major.width":  PLOT_STYLE["tick_linewidth"],
        "xtick.labelsize":    _fs("tick"),
        "ytick.labelsize":    _fs("tick"),
        "axes.titlesize":     _fs("title"),
        "pdf.fonttype":       42,
        "svg.fonttype":       "none",
    })


def _save_fig(fig, output_dir, stem):
    """
    Saves a figure as a PDF and then closes it to free memory.

    Why PDF instead of PNG?
    PDFs store drawings as mathematical shapes (vectors), not pixels.
    That means they stay perfectly sharp when zoomed in, printed, or
    embedded in a thesis — regardless of the display size.

    The dpi argument still matters for a PDF: any RASTER elements baked
    into the figure (e.g. scatter point markers, patheffect outlines) are
    rendered at this resolution before being embedded in the vector file.
    PLOT_STYLE["dpi_raster"] controls that resolution so it's consistent
    across every figure in the pipeline.

    Parameters:
        fig        : the matplotlib Figure object to save
        output_dir : folder path where the file will be written
        stem       : filename without extension (e.g. "Plot_Radius")
                     → saves as "Plot_Radius.pdf"
    """
    path = os.path.join(output_dir, stem + ".pdf")
    fig.savefig(path, bbox_inches="tight", dpi=PLOT_STYLE["dpi_raster"])
    plt.close(fig)  # Close the figure to free up memory

# ─────────────────────────────────────────────────────────────────────────────
# 1. CORE UNIFIED DRAWING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _darken(hex_color, factor=0.55):
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(hex_color))
    return tuple(rgb * factor)

def _draw_standard_violin_box(ax, groups, positions, colors, widths=0.10):
    """
    Central drawing engine enforcing identical violin and boxplot styles
    across all analysis modules. Excludes scatter elements entirely.
    """
    valid_indices = [i for i, g in enumerate(groups) if len(g) > 0]
    if not valid_indices:
        return

    v_groups = [groups[i] for i in valid_indices]
    v_positions = [positions[i] for i in valid_indices]
    v_colors = [colors[i] for i in valid_indices]
    v_dark = [_darken(c) for c in v_colors]

    v_idx_violin = [i for i, g in enumerate(v_groups) if len(g) >= 3 and np.std(g) > 0]
    if v_idx_violin:
        parts = ax.violinplot(
            [v_groups[i] for i in v_idx_violin],
            positions=[v_positions[i] for i in v_idx_violin],
            widths=0.70,
            showmedians=False,
            showextrema=False,
        )
        for body, col in zip(parts["bodies"], [v_colors[i] for i in v_idx_violin]):
            body.set_facecolor(col)
            body.set_edgecolor("#444444")
            body.set_alpha(0.80)
            body.set_linewidth(0.7)

    bx = ax.boxplot(
        v_groups,
        positions=v_positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        zorder=5,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color="#333333", linewidth=1.4),
        capprops=dict(color="#333333", linewidth=1.4),
        boxprops=dict(facecolor="#555555", edgecolor="#333333", linewidth=1.0),
    )

    for patch, dc in zip(bx["boxes"], v_dark):
        patch.set_facecolor(dc)
        patch.set_alpha(0.90)


def _clean_batch_labels(cond_df):
    """
    Replaces date-based batch labels (e.g. "260208 #1", "260316 #2") with
    clean, sequential numbering ("#1", "#2", "#3"...) for display purposes
    only, within a single condition's data.

    WHY THIS EXISTS
    ----------------
    Batch_Label is created upstream (analysis_batch.py / figure_assembly.py)
    in the format "<date> #<run>", e.g. "260208 #1". That's useful for
    tracing a vesicle back to its source experiment, but it's cluttered and
    hard to read as an axis label, especially once a condition has multiple
    dates (e.g. BranchedCortex pooling batches from both 260208 and 260316).

    This function does NOT modify the upstream Batch_ID/Batch_Label
    pipeline — it only relabels a COPY of this condition's data, right
    before plotting, so the fix is self-contained to plotting.py.

    HOW THE RENUMBERING WORKS
    --------------------------
    The distinct Batch_Label values for this condition are sorted (which
    sorts chronologically here, since the date prefix sorts correctly as a
    string), then renumbered #1, #2, #3... in that order. This guarantees
    unique, sequential labels even when multiple dates are pooled under one
    condition — e.g. "260208 #1", "260208 #2", "260316 #1" become
    "#1", "#2", "#3" rather than colliding on "#1" twice.

    Parameters
    ----------
    cond_df : pandas DataFrame, already filtered to ONE condition, with a
              'Batch_Label' column.

    Returns
    -------
    A copy of cond_df with 'Batch_Label' replaced by the clean sequential
    labels. If 'Batch_Label' is missing, returns the input unchanged
    (skip-and-warn pattern — never crashes the plotting pipeline).
    """
    if 'Batch_Label' not in cond_df.columns:
        return cond_df

    cond_df = cond_df.copy()
    ordered_originals = sorted(cond_df['Batch_Label'].unique())
    rename_map = {old: f"#{i + 1}" for i, old in enumerate(ordered_originals)}
    cond_df['Batch_Label'] = cond_df['Batch_Label'].map(rename_map)
    return cond_df

# ─────────────────────────────────────────────────────────────────────────────
# 2. SIZE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_violin_scatter_box(data, x_col, y_col, title, ylabel, output_dir, filename):
    set_paper_style()
    order = [c for c in CONDITION_ORDER if c in data[x_col].unique()]
    colors = [CONDITION_PALETTE[c] for c in order]
    # Median per condition, shown under each condition name instead of N.
    # np.nanmedian skips any NaNs so a few missing values can't silently
    # skew this (the same way .median() on a pandas Series would).
    medians = {c: np.nanmedian(data.loc[data[x_col] == c, y_col].values) for c in order}

    fig, ax = plt.subplots(figsize=_get_figsize_single_col())

    groups = [data.loc[data[x_col] == c, y_col].dropna().values for c in order]
    _draw_standard_violin_box(ax, groups, range(len(order)), colors, widths=0.10)

    # This metric (radius, or whatever else this function is called with)
    # can't be negative, so the axis should always start at exactly 0
    # rather than wherever matplotlib's autoscale margin happens to land.
    ax.set_ylim(bottom=0)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        # :.2f -> 2 decimal places (e.g. "Median=3.42"). Drop to :.1f if
        # that looks like more precision than the data actually supports.
        [f"{CONDITION_LABELS[c]}\nη={medians[c]:.2f} µm" for c in order],
        fontsize=_fs("tick"),
    )
    ax.set_ylabel(ylabel, fontsize=_fs("label"))
    #ax.set_title(title, fontsize=_fs("title"), pad=10)
    ax.set_xlim(-0.6, len(order) - 0.4)

    plt.tight_layout()
    stem = os.path.splitext(filename)[0]  # Strip extension so _save_fig adds .pdf
    _save_fig(fig, output_dir, stem)



def plot_superplot(data, x_col, y_col, batch_col, title, ylabel, output_dir, filename):
    """
    Generates a SuperPlot displaying individual vesicle data points in the background
    colored by condition class with distinct marker shapes per experiment/batch,
    overlaid with batch means, grand mean ± SEM, and category median (eta) annotations.
    """
    # List of distinct markers to cycle through for experiments/batches
    EXPERIMENT_MARKERS = ['o', 's', '^', '*', 'p', 'D', 'v', '<', '>', 'h']
    set_paper_style()
    order = [c for c in CONDITION_ORDER if c in data[x_col].unique()]
    
    # 1. Slightly widen figure dimensions to give columns more breathing room
    fig_width, fig_height = _get_figsize_single_col()
    fig, ax = plt.subplots(figsize=(fig_width * 1.15, fig_height))

    category_labels = []

    for x_idx, category in enumerate(order):
        cat_df = data[data[x_col] == category].dropna(subset=[y_col])
        base_label = CONDITION_LABELS.get(category, category)
        category_labels.append(base_label)

        if cat_df.empty:
            continue

        # Calculate overall category median for annotation
        cat_median = cat_df[y_col].median()
        
        # 2. Add median text directly under each column with a smaller font size
        ax.text(
            x_idx, -0.08, 
            f"$\\eta = {cat_median:.2f}~\\mu\\mathrm{{m}}$", 
            transform=ax.get_xaxis_transform(),
            ha='center', 
            va='top', 
            fontsize=_fs("tick") * 0.82
        )

        # Get defined color for this condition class
        class_color = CONDITION_PALETTE.get(category, '#333333')

        batches = cat_df[batch_col].unique() if (batch_col and batch_col in cat_df.columns) else [None]
        batch_means = []

        for b_idx, batch in enumerate(batches):
            marker = EXPERIMENT_MARKERS[b_idx % len(EXPERIMENT_MARKERS)]

            if batch is not None:
                b_df = cat_df[cat_df[batch_col] == batch]
            else:
                b_df = cat_df

            y_vals = b_df[y_col].values
            if len(y_vals) == 0:
                continue

            # Scatter individual vesicle points with jitter
            jitter = np.random.normal(0, 0.05, size=len(y_vals))
            ax.scatter(
                np.full_like(y_vals, x_idx) + jitter,
                y_vals,
                color=class_color,
                marker=marker,
                alpha=0.3,
                s=16,
                edgecolor='none',
                zorder=2
            )

            # Compute per-batch replicate mean
            b_mean = np.mean(y_vals)
            batch_means.append(b_mean)
            
            # Plot large batch mean marker
            ax.scatter(
                x_idx,
                b_mean,
                color=class_color,
                marker=marker,
                s=90,
                edgecolor='black',
                linewidth=1.2,
                zorder=4
            )

        # Overall Condition Grand Mean ± SEM error bar line
        if batch_means:
            grand_mean = np.mean(batch_means)
            sem = np.std(batch_means, ddof=1) / np.sqrt(len(batch_means)) if len(batch_means) > 1 else 0
            
            ax.errorbar(
                x_idx,
                grand_mean,
                yerr=sem,
                fmt='_',
                color='black',
                markersize=22,
                mew=2.5,
                capsize=6,
                capthick=2,
                zorder=5
            )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(category_labels, fontsize=_fs("tick"))
    ax.set_ylabel(ylabel, fontsize=_fs("label"))
    ax.set_ylim(bottom=0)
    
    # 3. Increase padding on x-axis limits to widen spacing
    ax.set_xlim(-0.7, len(order) - 0.3)
    
    sns.despine(ax=ax)

    plt.tight_layout()
    stem = os.path.splitext(filename)[0]
    _save_fig(fig, output_dir, stem)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PHENOTYPE COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def plot_phenotype_composition(df, x_col, phenotype_col, output_dir, filename, _as_figure=False):
    set_paper_style()
    order = [c for c in CONDITION_ORDER if c in df[x_col].unique()]
    pheno_order = ["Empty", "Lumenal", "Shell", "Sparse", "Patchy", "Continuous", "Excluded"]

    counts = df.groupby([x_col, phenotype_col]).size().unstack(fill_value=0).reindex(order)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pheno_present = [p for p in pheno_order if p in pct.columns]
    pct = pct[pheno_present]

    # Your chosen font sizes
    FS_LABEL = 26
    FS_TICK = 22
    FS_ANNOT = 25
    FS_LEGEND = 24

    fig = plt.figure(figsize=(6.0, 10.0))
    # Your chosen axes layout
    ax = fig.add_axes([0.18, 0.12, 1.1, 0.7])

    bottom = np.zeros(len(order))
    
    for pheno in pheno_present:
        vals = pct[pheno].values
        ax.bar(
            range(len(order)), vals, bottom=bottom, width=0.80,
            color=PHENOTYPE_PALETTE.get(pheno, "#aaaaaa"),
            edgecolor="#333333", linewidth=1.8, label=pheno
        )
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(j, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=FS_ANNOT, color="white", fontweight="bold",
                        path_effects=[pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()])
        bottom += vals

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in order], fontsize=FS_TICK, fontweight="bold")
    ax.set_xlim(-0.5, len(order) - 0.5)
    
    ax.set_ylabel("Percentage of GUVs (%)", fontsize=FS_LABEL, fontweight="bold")
    ax.tick_params(axis='y', labelsize=FS_TICK, width=1.8, length=8)
    ax.tick_params(axis='x', pad=8)
    ax.set_ylim(0, 108)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    handles = [mpatches.Patch(facecolor=PHENOTYPE_PALETTE.get(p, "#aaa"), edgecolor="#333333", linewidth=1.2, label=p) for p in pheno_present]
    
    legend = ax.legend(
        handles=handles, 
        loc="upper center", 
        # Tightened vertical offset for height=0.8 axes
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(4, len(pheno_present)), 
        fontsize=FS_LEGEND, 
        frameon=False, 
        title="Phenotype",
        columnspacing=0.5,   
        handletextpad=0.3,   
        handlelength=0.8     
    )
    if legend is not None:
        legend.get_title().set_fontsize(FS_LEGEND)
        legend.get_title().set_weight("bold")

    if _as_figure:
        return fig
    stem = os.path.splitext(filename)[0]
    _save_fig(fig, output_dir, stem)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PAIR PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_pairplot(cond_df, output_dir, filename_suffix="", _as_figure=False):
    from scipy.stats import spearmanr
    from matplotlib.patches import Patch
    set_paper_style()

    metrics = ["Refined Radius (um)", "t_cortex", "A localization", "Gini_Index"]
    cols_present = [c for c in metrics if c in cond_df.columns]
    plot_df = cond_df[cols_present + ["Phenotype_Category"]].dropna(subset=cols_present)
    plot_df = plot_df[plot_df["Phenotype_Category"] != "Empty"]

    if plot_df.empty:
        return

    # Master list of all possible phenotypes to ensure uniform legend width across all conditions
    pheno_order_all = ["Shell", "Sparse", "Patchy", "Continuous", "Excluded", "Lumenal"]
    phenotypes = plot_df["Phenotype_Category"].unique().tolist()
    phenotypes_sorted = [p for p in pheno_order_all if p in phenotypes]
    palette = {p: PHENOTYPE_PALETTE.get(p, "#aaaaaa") for p in phenotypes_sorted}

    plot_df = plot_df.rename(columns=METRIC_LABELS)
    cols_display = [METRIC_LABELS.get(c, c) for c in cols_present]

    # --- Typography Scale ---
    FS_TITLE    = 25
    FS_LABEL    = 25
    FS_TICK     = 25
    FS_ANNOT    = 25
    FS_CBAR_NUM = 25
    FS_LEGEND   = 25    

    def _corr_upper_closure(x, y, **kwargs):
        ax = plt.gca()
        if getattr(ax, '_corr_drawn', False):
            return
        ax._corr_drawn = True

        ax.set_xticks([])
        ax.set_yticks([])

        col_x, col_y = x.name, y.name
        full_x, full_y = plot_df[col_x], plot_df[col_y]

        mask = ~(np.isnan(full_x.values) | np.isnan(full_y.values))
        valid_x, valid_y = full_x.values[mask], full_y.values[mask]

        if len(valid_x) < 5 or np.std(valid_x) == 0 or np.std(valid_y) == 0:
            ax.set_facecolor("#e5e5e5")
            ax.text(0.5, 0.5, "n.d.", transform=ax.transAxes, ha="center", va="center",
                    fontsize=FS_ANNOT, color="#999999")
            return

        r, p = spearmanr(valid_x, valid_y)
        color = plt.cm.RdBu_r((r + 1) / 2)
        ax.set_facecolor(color)

        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = "white" if luminance < 0.55 else "#222222"
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        
        ax.text(0.5, 0.5, f"r={r:.2f}\n{stars}", transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=text_color)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        
        g = sns.PairGrid(
            plot_df, vars=cols_display, hue="Phenotype_Category",
            palette=palette, hue_order=phenotypes_sorted,
            diag_sharey=False
        )
        
        g.map_diag(sns.kdeplot, fill=True, alpha=0.45, linewidth=1.2, common_norm=False, warn_singular=False)
        g.map_lower(sns.scatterplot, s=16, alpha=0.50, edgecolor="none", linewidth=0)
        g.map_upper(_corr_upper_closure)

    # Increased figure width and height to comfortably accommodate font size 25 across 6 legend columns
    g.figure.set_size_inches(14.0, 13.5)
    
    # Adjusted margins to provide vertical and horizontal breathing room for large text
    g.figure.subplots_adjust(
        left   = 0.1300,  
        right  = 0.9500,  
        top    = 0.9000,  
        bottom = 0.2800,  
        wspace = 0.10, 
        hspace = 0.10
    )

    n_vars = len(cols_display)
    main_axes_ids = {id(a) for a in g.axes.flat}
    for ax in g.figure.axes:
        if id(ax) not in main_axes_ids:
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_yticks([])

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            ax = g.axes[i, j]
            for spine in ax.spines.values():
                spine.set_visible(False)

    Y_LABEL_OVERRIDE = {"GUV Radius (µm)": "Radius (µm)"}

    for ax in g.axes.flat:
        ax.tick_params(axis="both", width=1.1, length=4)
        
        plt.setp(ax.get_xticklabels(), fontsize=FS_TICK, fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontsize=FS_TICK, fontweight="bold")
        
        if ax.xaxis.label.get_text():
            ax.xaxis.label.set_fontsize(FS_LABEL)
            ax.xaxis.label.set_weight("bold")
            ax.xaxis.labelpad = 5
            
        if ax.yaxis.label.get_text():
            short = Y_LABEL_OVERRIDE.get(ax.yaxis.label.get_text())
            label_text = short if short else ax.yaxis.label.get_text()
            ax.set_ylabel(label_text, rotation=90)
            ax.yaxis.label.set_fontsize(FS_LABEL)
            ax.yaxis.label.set_weight("bold")
            ax.yaxis.labelpad = 12

    g.axes[-1, 0].set_xticks([])
    g.axes[-1, 0].set_yticks([])
    
    g.figure.canvas.draw()

    ax_x0 = g.axes[-1, 0].get_position().x0
    ax_x1 = g.axes[-1, -1].get_position().x1
    ax_width = ax_x1 - ax_x0

    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    
    # Position colorbar safely in the bottom margin space
    cax = g.figure.add_axes([ax_x0, 0.200, ax_width, 0.020])
    cbar = g.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Spearman ρ", fontsize=FS_LABEL, fontweight="bold", labelpad=6)
    plt.setp(cbar.ax.get_xticklabels(), fontsize=FS_CBAR_NUM, fontweight="bold")
    cbar.ax.tick_params(width=1.1, length=4)

    # Force a uniform static legend using the complete master list
    legend_handles = [
        Patch(facecolor=PHENOTYPE_PALETTE.get(p, "#aaaaaa"), label=p)
        for p in pheno_order_all
    ]
    
    # Position the 6-column legend further down below the colorbar
    g.figure.legend(
        handles        = legend_handles,
        title          = "Phenotype",
        fontsize       = FS_LEGEND,
        bbox_to_anchor = (ax_x0 + ax_width / 2, 0.035),
        loc            = "lower center",
        ncol           = len(pheno_order_all),
        markerscale    = 1.5,
        handletextpad  = 0.4,
        columnspacing  = 1.0,
        handlelength   = 1.2,
        borderaxespad  = 0.0,
    )
    if g.figure.legends:
        g.figure.legends[0].get_title().set_fontsize(FS_LEGEND)
        g.figure.legends[0].get_title().set_weight("bold")

    condition_label = filename_suffix.lstrip("_")
    g.figure.suptitle(
        f"Correlation Matrix & Pair Plot — {CONDITION_LABELS.get(condition_label, condition_label)}",
        y=0.945, fontsize=FS_TITLE, fontweight="bold"
    )

    stem = f"Correlation_Matrix_PairPlot{filename_suffix}"
    if _as_figure:
        return g.figure
    
    g.figure.savefig(os.path.join(output_dir, stem + ".pdf"), dpi=PLOT_STYLE["dpi_raster"], bbox_inches="tight")
    plt.close(g.figure)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PHENOTYPE CHARACTERISTICS MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def _add_sig_brackets(ax, df, metric, pheno_order, annot_fontsize=None, headroom_reserved=False, data_top=None):
    """
    Draws significance brackets between adjacent phenotype groups in the
    Phenotype Characteristics Matrix.

    Stars above the bracket = statistical significance (p-value).
    Bracket line thickness  = practical effect size (Cliff's Delta).
    No extra text is added — all information lives in the visual.

    Parameters
    ----------
    annot_fontsize : float, optional
        The matplotlib fontsize to use for the star annotation. Pass in
        a value from _fs_for_width("annot", <this figure's actual width>)
        so the stars print at the correct size regardless of how wide this
        particular figure was drawn. Falls back to _fs("annot") — the
        standard single-column size — if not provided.
    headroom_reserved : bool, optional
        Set True when the caller has ALREADY sized the axis's ylim to
        include room for these brackets (e.g. via _count_significant_pairs
        ahead of time). In that case this function draws the brackets
        within the existing space and does NOT call set_ylim() again —
        doing so would double-count the headroom and re-inflate the axis,
        recreating the exact "wasted empty space" problem this was meant
        to fix. Defaults to False so standalone callers keep their original
        self-expanding behavior.
    data_top : float, optional
        Only used when headroom_reserved=True. The y-value where the DATA's
        own range ends (e.g. the 99th-percentile cap), BEFORE any bracket
        headroom was added on top of it. Brackets stack starting from this
        value rather than from ax.get_ylim()[1], which — when headroom was
        pre-reserved — is already the inflated ceiling, not the data top.
    """
    if annot_fontsize is None:
        annot_fontsize = _fs("annot")

    pairs = [(pheno_order[i], pheno_order[i + 1]) for i in range(len(pheno_order) - 1)]
    sig = []  # Will hold (x1, x2, stars, linewidth, group_top) for each significant pair

    for p1, p2 in pairs:
        g1 = df.loc[df["Phenotype_Category"] == p1, metric].dropna().values
        g2 = df.loc[df["Phenotype_Category"] == p2, metric].dropna().values
        if len(g1) < 3 or len(g2) < 3:
            continue  # Skip if either group is too small to test reliably
        try:
            _, pval = mannwhitneyu(g1, g2, alternative="two-sided")
            if pval < 0.05:
                stars     = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
                delta     = _cliffs_delta(g1, g2)     # How big is the difference?
                lw        = _delta_linewidth(delta)    # Thick line = large effect
                # Use the TRUE max of each group, not the 95th percentile.
                # The violin's KDE shape is drawn out to the data's actual
                # min/max (plus a little extra from kernel smoothing at the
                # tails), so anchoring against the 95th percentile let the
                # bracket sit lower than the violin's visible top — the
                # bracket line ended up grazing or crossing the violin tail
                # instead of floating clearly above it.
                group_top = max(np.nanmax(g1), np.nanmax(g2))
                sig.append((pheno_order.index(p1), pheno_order.index(p2), stars, lw, group_top))
        except Exception:
            pass

    if not sig:
        return  # Nothing significant — nothing to draw

    all_vals = df[metric].dropna().values
    iqr = np.nanpercentile(all_vals, 75) - np.nanpercentile(all_vals, 25)
    if iqr < 1e-9:
        iqr = (all_vals.max() - all_vals.min()) * 0.1
    step = max(iqr * 0.55, np.nanpercentile(all_vals, 95) * 0.06)

    if headroom_reserved and data_top is not None:
        # The caller already expanded ylim to fit these brackets, starting
        # from data_top (the data's own ceiling, not the inflated axis top).
        # Stack brackets from THAT point — using ax.get_ylim()[1] here would
        # start stacking from the already-inflated ceiling and double the
        # reserved space.
        # The margin multiplier (0.3 -> 0.55) gives extra clearance above
        # whichever violin tail is tallest, since the KDE shape can bulge
        # slightly past the raw data max once smoothed.
        base_y = max(data_top, max(s[4] for s in sig)) + step * 0.55
    else:
        current_top = ax.get_ylim()[1]
        base_y = max(current_top, max(s[4] for s in sig)) + step * 0.55

    # Brackets connect category CENTERS (integer x-positions), but each
    # violin has its own width around that center (widths=0.70 in
    # _draw_standard_violin_box). Drawing the bracket flush between the two
    # centers makes it span the full category-to-category distance —
    # visually much longer than the actual gap between the two violin
    # shapes, especially for adjacent categories. Insetting each endpoint
    # pulls the line in to start/end just past each violin's edge instead.
    BRACKET_INSET = 0.22  # fraction of one category-spacing unit

    for i, (x1, x2, stars, lw, _) in enumerate(sig):
        y      = base_y + step * i
        tick_h = step * 0.15
        x1_drawn = x1 + BRACKET_INSET
        x2_drawn = x2 - BRACKET_INSET
        # Single flat horizontal bar — no vertical tick lines
        ax.plot([x1_drawn, x2_drawn], [y, y], color="black", lw=lw)
        # Stars annotation — just the stars, no delta text
        ax.text((x1 + x2) / 2, y + step * 0.05, stars,
                ha="center", va="bottom", fontsize=annot_fontsize, color="black")

    if not headroom_reserved:
        # Standalone behavior: expand the axis to fit what was just drawn.
        ax.set_ylim(ax.get_ylim()[0], base_y + step * (len(sig) + 0.8))
    else:
        # Even though the caller pre-reserved headroom for this row, that
        # estimate was based on the COMBINED data across every column
        # sharing this y-axis (sharey='row'). base_y above is computed from
        # THIS column's own true data max, which can occasionally exceed
        # the combined estimate — e.g. one condition's tail reaching
        # further than the row-level 99th-percentile used to size the
        # shared ceiling. When that happens, the topmost star gets drawn
        # right at (or past) the axis's existing top, leaving no margin
        # before the column title sitting just above it via `pad` — the
        # exact cause of stars overlapping the "Linear" title.
        #
        # This only expands when the pre-reserved headroom genuinely
        # wasn't enough for THIS column, so well-budgeted columns are
        # untouched and keep their original tight spacing.
        topmost_star_top = base_y + step * (len(sig) - 1 + 0.55)
        if topmost_star_top > ax.get_ylim()[1]:
            ax.set_ylim(ax.get_ylim()[0], topmost_star_top)

def _count_significant_pairs(df, metric, pheno_order):
    """
    Dry-run version of _add_sig_brackets: figures out how many adjacent
    phenotype pairs would be significant for this metric, WITHOUT drawing
    anything. Used to pre-compute how much extra headroom a row needs
    before any panel is actually plotted.

    WHY THIS EXISTS
    ----------------
    Previously, each column's headroom for stacked significance brackets
    was only known AFTER _add_sig_brackets() ran and called set_ylim() on
    that specific column's Axes. But sharey='row' means every column in
    that row inherits whichever column's ylim ended up tallest — so a
    column with 2 stacked "***" brackets could force a column with ZERO
    significant brackets to share that same tall y-range, leaving a large
    patch of dead space above its (otherwise normal-sized) violin.

    By counting brackets for every column FIRST, generate_category_panel
    can size the row's shared y-ceiling to the actual maximum needed,
    rather than discovering it column-by-column after the fact.

    Returns
    -------
    int : the number of significant adjacent pairs (0 if none, or if there
          isn't enough data to test reliably).
    """
    pairs = [(pheno_order[i], pheno_order[i + 1]) for i in range(len(pheno_order) - 1)]
    n_sig = 0
    for p1, p2 in pairs:
        g1 = df.loc[df["Phenotype_Category"] == p1, metric].dropna().values
        g2 = df.loc[df["Phenotype_Category"] == p2, metric].dropna().values
        if len(g1) < 3 or len(g2) < 3:
            continue
        try:
            _, pval = mannwhitneyu(g1, g2, alternative="two-sided")
            if pval < 0.05:
                n_sig += 1
        except Exception:
            pass
    return n_sig


def generate_category_panel(df, output_dir, _as_figure=False):
    set_paper_style()
    metrics = ["t_cortex", "A localization", "Gini_Index"]
    metric_lbls = METRIC_LABELS
    cond_order = [c for c in CONDITION_ORDER if c in df["Category"].unique() and c != "Empty"]
    
    pheno_all = ["Shell", "Sparse", "Patchy", "Continuous"]
    cond_phenos = {}
    
    # Establish valid phenotypes per condition so we can ratio subplot widths properly
    for cond in cond_order:
        cond_df = df[df["Category"] == cond]
        cond_phenos[cond] = [p for p in pheno_all if p in cond_df["Phenotype_Category"].unique()]

    # Filter out empty conditions entirely
    cond_order = [c for c in cond_order if cond_phenos[c]]
    if not cond_order:
        return

    # Physical scaling ratios to enforce uniformly wide violins
    width_ratios = [len(cond_phenos[c]) for c in cond_order]
    total_cats = sum(width_ratios)
    # NOTE: this used to include a "* 1.8 for A5 scaling" multiplier, left
    # over from before _fs_for_width() existed. Back then, fonts didn't
    # scale with canvas width, so the canvas itself was inflated by hand to
    # compensate. Now that _fs_for_width() computes the correct matplotlib
    # fontsize FROM the actual canvas width, that multiplier is redundant —
    # worse, it inflated this figure to ~25in wide, which made
    # _fs_for_width() compute absurdly large fonts (~50pt) to compensate for
    # the (unnecessary) extra shrinkage. Those oversized labels no longer
    # fit the row height, which is what caused the y-axis labels from
    # different rows to visually overlap each other.
    fig_width = 1.35 * total_cats + 0.8 * len(cond_order) * 1.4

    fig, axes = plt.subplots(
        len(metrics), len(cond_order),
        figsize=(fig_width, 3.5 * len(metrics)),
        sharey='row',
        gridspec_kw={'width_ratios': width_ratios},
        squeeze=False
    )

    # ── Pre-compute a sensible y-axis ceiling for each metric ─────────────────
    # The default auto-scaling stretches to include violin tail outliers, which
    # pushes the y-axis far above where the bulk of the data lives.
    # Instead we cap at the 99th percentile of each metric across all relevant
    # conditions, then add headroom for labels AND for however many
    # significance brackets the busiest column in this row actually needs.
    #
    # WHY WE COUNT BRACKETS HERE (not just add a flat margin):
    # sharey='row' means every column in a row shares one y-axis. Without
    # this step, the column with the most stacked "***" brackets would push
    # the ylim up for ALL columns in that row — including ones with zero
    # significant brackets — leaving large empty gaps above their violins.
    # By finding the max bracket count up front, the headroom added here
    # is exactly what the busiest column needs, and no more.
    relevant_df = df[df["Category"].isin(cond_order)]
    y_caps = {}
    data_tops = {}  # The data's own ceiling, BEFORE bracket headroom was added — needed by _add_sig_brackets so it stacks from the right starting point.
    for metric in metrics:
        vals = relevant_df[metric].dropna()
        if len(vals) == 0:
            y_caps[metric] = None
            data_tops[metric] = None
            continue

        p99 = np.nanpercentile(vals, 99)
        base_top = p99 * 1.15  # 15% headroom above the 99th percentile
        data_tops[metric] = base_top

        # How many brackets does the busiest column in this row need?
        max_sig_pairs = 0
        for cond in cond_order:
            cond_df = df[df["Category"] == cond]
            pheno_present = cond_phenos[cond]
            sub = cond_df[["Phenotype_Category", metric]].dropna(subset=[metric])
            if sub.empty:
                continue
            n_sig = _count_significant_pairs(sub, metric, pheno_present)
            max_sig_pairs = max(max_sig_pairs, n_sig)

        if max_sig_pairs == 0:
            y_caps[metric] = base_top
        else:
            # Mirrors _add_sig_brackets' own step/spacing formula, so the
            # headroom reserved here matches what it will actually use.
            # The +1.35 buffer (was +1.1) matches the increased per-bracket
            # margin in _add_sig_brackets (0.3 -> 0.55), keeping this
            # pre-allocation and the actual drawing in sync.
            iqr = np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)
            if iqr < 1e-9:
                iqr = (vals.max() - vals.min()) * 0.1
            step = max(iqr * 0.55, np.nanpercentile(vals, 95) * 0.06)
            y_caps[metric] = base_top + step * (max_sig_pairs + 1.35)

    for ci, cond in enumerate(cond_order):
        cond_df = df[df["Category"] == cond]
        pheno_present = cond_phenos[cond]

        for ri, metric in enumerate(metrics):
            ax = axes[ri, ci]
            sub = cond_df[["Phenotype_Category", metric]].dropna(subset=[metric])

            if sub.empty:
                ax.set_visible(False)
                continue

            groups = [sub.loc[sub["Phenotype_Category"] == p, metric].values for p in pheno_present]
            positions = list(range(len(pheno_present)))
            colors = [PHENOTYPE_PALETTE.get(p, "#aaaaaa") for p in pheno_present]

            _draw_standard_violin_box(ax, groups, positions, colors, widths=0.12)

            # Set y limits: bottom anchored at 0, top capped at 99th-percentile ceiling.
            # sharey='row' means this limit is shared across all columns in the row,
            # so the tightest sensible top is the right value to use here.
            top = y_caps.get(metric)
            if top is not None:
                ax.set_ylim(bottom=0, top=top)
            else:
                ax.set_ylim(bottom=0)
            ax.set_xlim(-0.6, len(pheno_present) - 0.4)
            
            _add_sig_brackets(ax, sub, metric, pheno_present, annot_fontsize=_fs_for_width("annot", fig_width),
                               headroom_reserved=True, data_top=data_tops.get(metric))

            ax.set_xticks(positions)

            # ── FIX: y-axis tick numbers were falling back to the global
            # rcParam default (calibrated for a 7in-wide single-column
            # figure) instead of scaling with THIS figure's actual width
            # (~11.5in for a 3-condition grid). That mismatch is what made
            # the y-axis numbers (0.0, 2.5, 5.0...) print visibly smaller
            # than the x-axis phenotype labels right next to them. This
            # line sizes them with the same fig_width-aware formula the
            # x-labels already use below, so both match.
            ax.tick_params(axis='y', labelsize=_fs_for_width("tick", fig_width))

            # Only show x-axis labels on the bottom row.
            # For all other rows, tick marks stay (for visual alignment) but
            # the text labels are hidden — no need to repeat them every row.
            if ri == len(metrics) - 1:
                ax.set_xticklabels(pheno_present, fontsize=_fs_for_width("tick", fig_width), rotation=20)
            else:
                ax.tick_params(labelbottom=False)  # Hide text, keep tick marks

            # Show the condition name as a column title on the top row only.
            # Now that x-labels only appear once at the bottom, the reader needs
            # the column header to know which condition each column represents.
            #
            # pad is scaled relative to the annotation fontsize (not a fixed
            # number) because _add_sig_brackets can stack multiple brackets
            # above the data, pushing the axes' effective top higher when
            # there are more significant pairs. A fixed small pad let the
            # title collide with the topmost bracket's stars; this keeps a
            # consistent visual gap regardless of how many brackets stack up.
            if ri == 0:
                ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=_fs_for_width("title", fig_width),
                             fontweight='bold', pad=_fs_for_width("annot", fig_width) * 2.2)

            if ci == 0:
                ax.set_ylabel(metric_lbls.get(metric, metric), fontsize=_fs_for_width("label", fig_width))
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Phenotype_Characteristics_Matrix_Stats")

# ─────────────────────────────────────────────────────────────────────────────
# 6. BATCH SIZE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_batch_size_distribution(df, output_dir, _as_figure=False):
    set_paper_style()
    metric = 'Refined Radius (um)'
    if metric not in df.columns: return

    conditions = [c for c in CONDITION_ORDER if c in df['Category'].unique()]
    ncols = min(2, len(conditions))
    nrows = int(np.ceil(len(conditions) / ncols))
    # Widened from 4.5in: with conditions that pool batches from multiple
    # dates (up to ~5 batches in one panel), the N-labels need more room
    # to avoid overlapping each other.
    panel_size_in = 5.8
    fig_width = ncols * panel_size_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, nrows * panel_size_in), squeeze=False)
    ax_flat = axes.flatten()

    for idx, condition in enumerate(conditions):
        ax = ax_flat[idx]
        cond_df = df[df['Category'] == condition].copy()
        cond_df = _clean_batch_labels(cond_df)  # "260208 #1" -> "#1", sequential per condition
        batch_order = sorted(cond_df['Batch_Label'].unique())
        cond_color = CONDITION_PALETTE.get(condition, '#6B7B8C')
        
        sub = cond_df[['Batch_Label', metric]].dropna(subset=[metric])
        if sub.empty:
            ax.set_visible(False)
            continue

        groups = [sub.loc[sub['Batch_Label'] == b, metric].values for b in batch_order]
        colors = [cond_color] * len(batch_order)
        
        _draw_standard_violin_box(ax, groups, range(len(batch_order)), colors, widths=0.12)

        # GUV radius can never be negative, so the axis should always start
        # at 0 rather than wherever matplotlib's autoscale margin happens to
        # land. Setting only `bottom=` here (not `top=`) is what lets the
        # later `set_ylim(top=...)` calls below adjust headroom without
        # undoing this.
        ax.set_ylim(bottom=0)

        overall_median = sub[metric].median()
        ax.axhline(overall_median, color='#2d7fb8', linestyle='--', linewidth=1.1, alpha=0.7)

        y_max = sub[metric].max()
        y_range = sub[metric].max() - sub[metric].min()
        for x_pos, batch in enumerate(batch_order):
            n_ves = sub[sub['Batch_Label'] == batch][metric].count()
            # Rotated 90 deg: a vertical label takes far less horizontal
            # room per batch than a horizontal one, which is what was
            # causing N-labels to run into each other when a panel has
            # several batches side by side.
            #
            # Anchored to the TRUE max (not the 99th percentile) with a
            # wider margin: violinplot's KDE is drawn out to the data's
            # actual max, and on right-skewed radius data (every condition
            # here has skewness > 0.5) the 99th percentile sits noticeably
            # below that, which let the label collide with the violin's
            # tapering tip.
            ax.text(x_pos, y_max + y_range * 0.08, f'N={n_ves}', ha='center', va='bottom',
                    rotation=90, fontsize=_fs_for_width("annot", fig_width), color='#555555')
        # Rotated text extends further upward than horizontal text would at
        # the same starting y-position, so reserve extra headroom above the
        # data — otherwise the N-label collides with the title sitting just
        # above the axes. Raised from 0.55: anchoring to the true max (see
        # above) starts the label higher than the old 99th-percentile
        # anchor did, which ate into this margin and let the tallest-N
        # label collide with the title again.
        ax.set_ylim(top=ax.get_ylim()[1] + y_range * 0.80)

        ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=_fs_for_width("title", fig_width), fontweight='bold', pad=14)
        ax.set_ylabel('GUV Radius (µm)', fontsize=_fs_for_width("label", fig_width))
        # Show x-labels only on the bottom row of the grid
        is_bottom_row = (idx // ncols) == (nrows - 1) or (idx + ncols) >= len(conditions)
        if is_bottom_row:
            ax.set_xticks(range(len(batch_order)))
            ax.set_xticklabels(batch_order, fontsize=_fs_for_width("tick", fig_width), rotation=30, ha='right')
        else:
            ax.set_xticks(range(len(batch_order)))
            ax.tick_params(labelbottom=False)

    for idx in range(len(conditions), len(ax_flat)): ax_flat[idx].set_visible(False)

    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Batch_Size_Distribution_All_Conditions")

# ─────────────────────────────────────────────────────────────────────────────
# 7. BATCH PHENOTYPE COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def plot_batch_phenotype_composition(df, output_dir, _as_figure=False):
    # --- ADJUST FONT SIZES HERE EASILY ---
    FS_TITLE  = 25
    FS_LABEL  = 25
    FS_TICK   = 25
    FS_ANNOT  = 25
    FS_LEGEND = 25

    set_paper_style()
    if 'Phenotype_Category' not in df.columns: return

    conditions = [c for c in CONDITION_ORDER if c in df['Category'].unique() and c != 'Empty']
    if not conditions: return

    pheno_order_all = ["Empty", "Lumenal", "Shell", "Sparse", "Patchy", "Continuous", "Excluded"]
    ncols = min(2, len(conditions))
    nrows = int(np.ceil((len(conditions) + 1) / ncols))  # +1 to make room for the legend slot if needed
    
    panel_w_in, panel_h_in = 5.5, 4.8
    fig_width = ncols * panel_w_in

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, nrows * panel_h_in), squeeze=False)
    ax_flat = axes.flatten()

    for idx, condition in enumerate(conditions):
        ax = ax_flat[idx]
        cond_df = df[df['Category'] == condition].copy()
        cond_df = _clean_batch_labels(cond_df)  # "260208 #1" -> "#1", sequential per condition
        batch_order = sorted(cond_df['Batch_Label'].unique())
        phenos_present = [p for p in pheno_order_all if p in cond_df['Phenotype_Category'].unique()]

        if not phenos_present or not batch_order:
            ax.set_visible(False)
            continue

        counts = cond_df.groupby(['Batch_Label', 'Phenotype_Category']).size().unstack(fill_value=0).reindex(index=batch_order, columns=phenos_present, fill_value=0)
        pct = counts.div(counts.sum(axis=1), axis=0) * 100

        bottom = np.zeros(len(batch_order))
        
        for pheno in phenos_present:
            if pheno not in pct.columns: continue
            vals = pct[pheno].values
            ax.bar(range(len(batch_order)), vals, bottom=bottom, color=PHENOTYPE_PALETTE.get(pheno, '#aaaaaa'), edgecolor='#333333', linewidth=0.8, label=pheno)
            # Annotate segments that are wide enough to read (> 6 %)
            for x_pos, (val, bot) in enumerate(zip(vals, bottom)):
                if val > 6:
                    ax.text(x_pos, bot + val / 2, f'{val:.0f}%',
                            ha='center', va='center',
                            fontsize=FS_ANNOT - 2, color='white', fontweight='bold',
                            path_effects=[pe.Stroke(linewidth=0.5, foreground='black'), pe.Normal()])
            bottom += vals

        for x_pos, batch in enumerate(batch_order):
            n_ves = len(cond_df[cond_df['Batch_Label'] == batch])
            ax.text(x_pos, 104, f'N={n_ves}', ha='center', va='bottom',
                    rotation=90, fontsize=FS_ANNOT, color='#555555', fontweight='bold')

        ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=FS_TITLE, fontweight='bold', pad=20)
        ax.set_ylabel('GUVs (%)', fontsize=FS_LABEL, fontweight='bold')
        
        ax.set_ylim(0, 145)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis='y', labelsize=FS_TICK)
        for label in ax.get_yticklabels():
            label.set_fontsize(FS_TICK)
            label.set_fontweight('bold')

        ax.set_xticks(range(len(batch_order)))
        ax.set_xticklabels(batch_order, fontsize=FS_TICK, rotation=30, ha='right', fontweight='bold')
        sns.despine(ax=ax)

    # --- PLACE LEGEND IN THE 4TH (UNUSED) GRID SLOT ---
    legend_ax_idx = len(conditions)
    if legend_ax_idx < len(ax_flat):
        legend_ax = ax_flat[legend_ax_idx]
        legend_ax.axis('off')  # Hide borders and ticks for the legend panel
        
        all_phenos_shown = [p for p in pheno_order_all if p in df['Phenotype_Category'].unique() and p != 'Empty']
        legend_patches = [mpatches.Patch(facecolor=PHENOTYPE_PALETTE.get(p, '#aaa'), edgecolor='#333333', linewidth=0.8, label=p) for p in all_phenos_shown]
        
        leg = legend_ax.legend(
            handles=legend_patches, loc='center', fontsize=FS_LEGEND, 
            frameon=True, facecolor='white', edgecolor='none', title='Phenotype'
        )
        leg.get_title().set_fontsize(FS_LEGEND)
        leg.get_title().set_fontweight('bold')

    # Hide any remaining surplus axes beyond the legend slot
    for idx in range(legend_ax_idx + 1, len(ax_flat)):
        ax_flat[idx].set_visible(False)

    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Batch_Phenotype_Composition_All_Conditions")

# ─────────────────────────────────────────────────────────────────────────────
# 8. BATCH ACTIN METRICS
# ─────────────────────────────────────────────────────────────────────────────

def plot_batch_actin_metrics(df, output_dir, _as_figure=False):
    set_paper_style()
    metrics = ['t_cortex', 'A localization', 'Gini_Index'] 
    metric_labels = METRIC_LABELS
    
    actin_conditions = [c for c in CONDITION_ORDER if c in df['Category'].unique() and c != 'Empty']
    if not actin_conditions: return

    metrics = [m for m in metrics if m in df.columns]
    if not metrics: return

    # Each panel needs enough width to fit up to ~5 batch columns with
    # N-labels above each one without them colliding (the previous 2.8in/
    # panel was sized for ~3 batches and crowded badly once a condition
    # pooled batches from multiple dates, e.g. 5 batches in one panel).
    fig_width = len(actin_conditions) * 4.2
    fig, axes = plt.subplots(len(metrics), len(actin_conditions), figsize=(fig_width, len(metrics) * 4.4), squeeze=False)

    for col_idx, condition in enumerate(actin_conditions):
        cond_df = df[df['Category'] == condition].copy()
        cond_df = _clean_batch_labels(cond_df)  # "260208 #1" -> "#1", sequential per condition
        batch_order = sorted(cond_df['Batch_Label'].unique())
        cond_color = CONDITION_PALETTE.get(condition, '#6B7B8C')

        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            sub = cond_df[['Batch_Label', metric]].dropna(subset=[metric]).copy()

            if sub.empty:
                ax.set_visible(False)
                continue

            groups = [sub.loc[sub['Batch_Label'] == b, metric].values for b in batch_order]
            colors = [cond_color] * len(batch_order)
            
            _draw_standard_violin_box(ax, groups, range(len(batch_order)), colors, widths=0.12)

            # t_cortex, A_loc, and Gini are all non-negative metrics, so the
            # axis should always start at 0 rather than wherever autoscale's
            # margin happens to land. Only `bottom=` is set here so the
            # row-0 headroom adjustment below (`top=...`) still works.
            ax.set_ylim(bottom=0)

            overall_median = sub[metric].median()
            ax.axhline(overall_median, color='#2d7fb8', linestyle='--', linewidth=1.0, alpha=0.65)

            if row_idx == 0: ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=_fs_for_width("title", fig_width), fontweight='bold', pad=14)
            if col_idx == 0: ax.set_ylabel(metric_labels.get(metric, metric), fontsize=_fs_for_width("label", fig_width))

            # Show x-labels only on the bottom metric row
            if row_idx == len(metrics) - 1:
                ax.set_xticks(range(len(batch_order)))
                ax.set_xticklabels(batch_order, fontsize=_fs_for_width("tick", fig_width), rotation=30, ha='right')
            else:
                ax.set_xticks(range(len(batch_order)))
                ax.tick_params(labelbottom=False)

            # Anchored to the TRUE max (see plot_batch_size_distribution for
            # why the 99th percentile let this collide with the violin tip).
            y_top = sub[metric].max()
            y_range = sub[metric].max() - sub[metric].min()
            for x_pos, batch in enumerate(batch_order):
                n_ves = sub[sub['Batch_Label'] == batch][metric].count()
                ax.text(x_pos, y_top + y_range * 0.08, f'N={n_ves}', ha='center', va='bottom',
                        rotation=90, fontsize=_fs_for_width("annot", fig_width), color='#555555')
            if row_idx == 0:
                # Only the top row has a title directly above it, so only
                # the top row needs the extra headroom to keep the rotated
                # N-label clear of the title text. Raised from 0.55 -- see
                # plot_batch_size_distribution for why.
                ax.set_ylim(top=ax.get_ylim()[1] + y_range * 0.80)

    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Batch_Actin_Metrics_All_Conditions")

# ─────────────────────────────────────────────────────────────────────────────
# 9. CROSS-CONDITION COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _mw_stars(vals_a, vals_b):
    """
    Runs a Mann-Whitney U test and returns a star-rating plus the raw p-value.

    Think of this like a "surprise meter" — it tells you how unlikely the
    observed difference is if the two groups were actually identical.

    Returns:
        stars (str): "***", "**", "*", or "ns" (not significant)
        p     (float): the actual p-value
    """
    if len(vals_a) < 5 or len(vals_b) < 5:
        return "ns", 1.0  # Not enough data to run the test reliably
    try:
        _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        return stars, p
    except Exception:
        return "ns", 1.0

def _cliffs_delta(vals_a, vals_b):
    """
    Calculates Cliff's Delta — a measure of *how big* the difference is,
    not just whether it exists.

    Imagine you randomly pick one vesicle from Group A and one from Group B.
    Cliff's Delta tells you: how often does the A-vesicle have a higher value
    than the B-vesicle, minus how often it's lower?

    The result ranges from -1 to +1:
      +1 means A is always larger than B
      -1 means B is always larger than A
       0 means they're essentially the same

    This is calculated by comparing every possible pair (one from A, one from B).
    It's like a round-robin tournament — every pair plays once.
    """
    n_a = len(vals_a)
    n_b = len(vals_b)

    if n_a == 0 or n_b == 0:
        return 0.0  # No data to compare

    # Count dominance: +1 if a > b, -1 if a < b, 0 if tied
    # We use numpy broadcasting to compare all n_a × n_b pairs at once
    # (faster than a nested for-loop for large datasets)
    a_col = vals_a[:, np.newaxis]   # Reshape A into a column  → shape (n_a, 1)
    b_row = vals_b[np.newaxis, :]   # Reshape B into a row     → shape (1, n_b)

    # Each cell in the resulting (n_a × n_b) matrix is one pair comparison
    dominance_matrix = np.sign(a_col - b_row)  # +1, -1, or 0

    delta = dominance_matrix.sum() / (n_a * n_b)  # Normalise to [-1, +1]
    return float(delta)

def _delta_label(delta):
    """
    Converts a Cliff's Delta number into a human-readable category string.

    Not used directly in plots (effect size is encoded as line thickness instead),
    but kept here as a utility for printed reports or CSV exports if needed.

    Thresholds (Romano et al., 2006):
      |δ| < 0.147  → "neg" (negligible)
      |δ| < 0.330  → "sml" (small)
      |δ| < 0.474  → "med" (medium)
      |δ| ≥ 0.474  → "lrg" (large)
    """
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "neg"
    elif abs_d < 0.330:
        return "sml"
    elif abs_d < 0.474:
        return "med"
    else:
        return "lrg"

def _delta_linewidth(delta):
    """
    Converts a Cliff's Delta value into a bracket line width.

    This encodes the *practical effect size* visually — thicker bracket = bigger
    real-world difference — without adding any extra text to the plot.

    Think of it like the weight of a pen stroke: a faint pencil line means
    "technically different but barely," while a bold marker means "clearly
    and substantially different."

    WHY THE FLOOR WAS RAISED
    --------------------------
    The original tiers (0.8 / 1.4 / 2.2 / 3.2 px) were designed on-screen,
    where a 0.8px line is still crisp. But once a figure is shrunk down to
    its real printed size on the A5 page (~5in wide, often less for a grid
    panel), anti-aliasing and the shrink factor together can make a sub-1px
    line nearly disappear — so a result that IS statistically significant
    (a star is shown) can read as having no visible bracket at all. That
    defeats the purpose of showing it.

    The tiers below keep the same relative step-up between negligible →
    small → medium → large (so "thicker = bigger effect" still reads
    correctly), but raise the floor so even "negligible" stays a clearly
    visible solid line at print scale.

    Mapping (Romano et al. thresholds):
        |δ| < 0.147  → 1.5 px  (negligible — visible, but clearly thinnest)
        |δ| < 0.330  → 2.5 px  (small)
        |δ| < 0.474  → 3.5 px  (medium)
        |δ| ≥ 0.474  → 4.8 px  (large — unmistakable)
    """
    abs_d = abs(delta)
    if abs_d < 0.147:
        return 1.5
    elif abs_d < 0.330:
        return 2.5
    elif abs_d < 0.474:
        return 3.5
    else:
        return 4.8

def _build_stat_label(vals_a, vals_b):
    """
    Runs the Mann-Whitney test and computes Cliff's Delta, then packages
    everything needed to draw an annotated bracket.

    The bracket text stays as plain stars (e.g. "***") — no extra text.
    The effect size is encoded silently through the bracket line width,
    which is determined here and passed to _add_bracket.

    Returns:
        stars (str):  "***", "**", "*", or "ns"
        label (str):  the stars string (used as the bracket annotation),
                      or None if not significant
        lw    (float): bracket line width encoding Cliff's Delta magnitude
    """
    stars, _ = _mw_stars(vals_a, vals_b)

    if stars == "ns":
        return "ns", None, 0.8  # Not significant — no bracket will be drawn

    delta = _cliffs_delta(vals_a, vals_b)
    lw    = _delta_linewidth(delta)  # Thick line = large effect, thin = small

    return stars, stars, lw  # label is just the stars — clean and uncluttered

def _add_bracket(ax, x1, x2, y_top, label, lw=0.9, color='black', fontsize=None, inset=0.13):
    """
    Draws a significance bracket between two groups on the plot.

    The stars annotation tells you *if* the difference is significant (p-value).
    The line thickness tells you *how big* the difference is (Cliff's Delta).
    No extra text is added — all the information lives in the visual itself.

    Parameters:
        ax       : the matplotlib Axes object to draw on
        x1, x2   : the x-positions of the two groups being compared
        y_top    : the data y-value just above the tallest bar (bracket root)
        label    : the star string, e.g. "***"
        lw       : line width encoding effect size (from _delta_linewidth)
        color    : line and text colour (default black)
        fontsize : matplotlib fontsize for the star text. Pass in
                   _fs_for_width("annot", <this figure's actual width>) so
                   the stars print at the right size for that figure's
                   canvas. Falls back to _fs("annot") if not provided.
        inset    : how far to pull each endpoint in from x1/x2 (in x-axis
                   data units -- categories here are always spaced 1.0
                   apart), so the bracket reads as connecting the two
                   violins rather than spanning the full category-to-
                   category distance edge-to-edge. Default pulls in 0.13
                   units from each side (~0.26 shorter overall).
    """
    if fontsize is None:
        fontsize = _fs("annot")

    x1_drawn = x1 + inset if x2 > x1 else x1 - inset
    x2_drawn = x2 - inset if x2 > x1 else x2 + inset

    # Scale all offsets to the current y-axis range so the bracket
    # looks proportional regardless of whether the metric is 0–1 or 0–50.
    # These are deliberately generous: violinplot's KDE can bulge slightly
    # past the raw data max it was anchored to, and the line+text need
    # enough combined headroom that stacked brackets (multiple calls at
    # increasing y_top) don't run into each other -- see the callers, which
    # space consecutive levels using their own `step`.
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_line  = y_top + (y_range * 0.08)   # The horizontal bar sits here
    y_text  = y_top + (y_range * 0.13)   # The stars annotation sits here

    # Draw the bracket: a single flat horizontal bar (no vertical tick lines)
    ax.plot([x1_drawn, x2_drawn], [y_line, y_line], lw=lw, color=color)

    # Write the star annotation above the bracket
    ax.text(
        (x1 + x2) / 2, y_text, label,
        ha='center', va='bottom',
        fontsize=fontsize, color=color,
    )

    # Expand the y-axis if the annotation would be clipped off the top
    needed_top = y_text + (y_range * 0.08)
    if needed_top > ax.get_ylim()[1]:
        ax.set_ylim(ax.get_ylim()[0], needed_top)

def _violin_box_comp(ax, data, x_col, y_col, order, palette):
    groups = [data.loc[data[x_col] == c, y_col].dropna().values for c in order]
    colors = [palette.get(c, "#aaaaaa") for c in order]
    _draw_standard_violin_box(ax, groups, range(len(order)), colors, widths=0.12)

def _comp_palette(order):
    return {c: CONDITION_PALETTE[c] for c in order if c in CONDITION_PALETTE}

def _fmt_xtick(ax, order, rotation=15):
    """
    Replaces raw column names (e.g. 'BranchedCortex') with readable labels
    (e.g. 'Branched Cortex') on the x-axis of a comparison plot.
    """
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in order],
        rotation=rotation, ha='right', fontsize=_fs("tick"),
    )

def plot_radius_all_conditions(df, output_dir):
    set_paper_style()
    metric = 'Refined Radius (um)'
    if metric not in df.columns: return

    order = [c for c in CONDITION_ORDER if c in df['Category'].unique()]
    palette = _comp_palette(order)
    plot_df = df[df['Category'].isin(order)].dropna(subset=[metric])

    fig, ax = plt.subplots(figsize=_get_figsize_single_col())
    _violin_box_comp(ax, plot_df, 'Category', metric, order, palette)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], rotation=15, ha='right', fontsize=_fs("tick"))
    ax.set_ylabel("GUV Radius (µm)", fontsize=_fs("label"))
    ax.set_ylim(bottom=0)  # Radius can't be negative — always start the axis at 0
    #ax.set_title("GUV Radius Across Conditions", fontsize=_fs("title"), pad=10)
    sns.despine()
    
    plt.tight_layout()
    _save_fig(fig, output_dir, "Plot_Radius_AllConditions")

def plot_cortex_vs_empty_radius(df, output_dir, _as_figure=False):
    set_paper_style()
    metric = 'Refined Radius (um)'
    if metric not in df.columns: return

    order = [c for c in CONDITION_ORDER if c in df['Category'].unique()]
    palette = _comp_palette(order)
    plot_df = df[df['Category'].isin(order)].dropna(subset=[metric])

    fig, ax = plt.subplots(figsize=_get_figsize_single_col())
    _violin_box_comp(ax, plot_df, 'Category', metric, order, palette)
 
    _fmt_xtick(ax, order)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], rotation=15, ha='right', fontsize=_fs("tick"))
    ax.set_ylabel("GUV Radius (µm)", fontsize=_fs("label"))
    ax.set_ylim(bottom=0)  # Radius can't be negative — always start the axis at 0
    #ax.set_title("GUV Radius: Cortex vs Empty", fontsize=_fs("title"), pad=10)
    sns.despine()
    
    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Plot_Cortex_vs_Empty_Radius")

def plot_actin_three_way(df, output_dir, _as_figure=False):
    set_paper_style()
    order = [c for c in ['Factin', 'BranchedCortex', 'LinearCortex'] if c in df['Category'].unique()]
    palette = _comp_palette(order)
    plot_df = df[df['Category'].isin(order)].copy()
    metrics = [m for m in ['A localization', 'Gini_Index', 't_cortex'] if m in plot_df.columns]

    if not metrics: return

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig_width = ncols * 3.8
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows * 4.2))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    metric_labels = METRIC_LABELS

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        m_df = plot_df.dropna(subset=[metric])

        _violin_box_comp(ax, m_df, 'Category', metric, order, palette)
        # These actin metrics are all non-negative — always start the axis at 0
        ax.set_ylim(bottom=0)

        try:
            x_pos = {c: i for i, c in enumerate(order)}
            # Step needs to clear _add_bracket's own per-level footprint
            # (line + text sit roughly 0.08-0.13*axis_range above y_top) with
            # margin to spare, or stacked levels' stars collide with the
            # next level's line -- 0.12 wasn't enough margin (confirmed:
            # the Gini panel's stacked stars were visibly overlapping).
            y_range = m_df[metric].max() - m_df[metric].min()
            step = y_range * 0.20
            global_max = m_df[metric].max()

            pairs = [('Factin', 'BranchedCortex', 0), ('Factin', 'LinearCortex', 1), ('BranchedCortex', 'LinearCortex', 2)]
            for cond_a, cond_b, level in pairs:
                if cond_a not in x_pos or cond_b not in x_pos: continue
                
                v_a = m_df[m_df['Category'] == cond_a][metric].dropna().values
                v_b = m_df[m_df['Category'] == cond_b][metric].dropna().values

                # A bracket spanning two NON-adjacent x-positions (e.g.
                # Factin-to-Linear, skipping over Branched) is drawn right
                # over whichever group sits in between -- so it must clear
                # THAT group's violin too, not just the two it's
                # statistically comparing. Anchoring to just v_a/v_b's max
                # let the skipped-over group's tail poke through the line
                # whenever it was taller than both compared groups.
                spans_middle = abs(x_pos[cond_a] - x_pos[cond_b]) > 1
                y_max = global_max if spans_middle else max(v_a.max(), v_b.max())

                # stars = significance level, label = stars string,
                # lw = line width encoding Cliff's Delta (thick = large effect)
                stars, label, lw = _build_stat_label(v_a, v_b)
                if stars != "ns":
                    # level * step ensures brackets stack vertically
                    _add_bracket(ax, x_pos[cond_a], x_pos[cond_b], y_max + (level * step), label, lw=lw,
                                 fontsize=_fs_for_width("annot", fig_width))
        except Exception: 
            pass

        # Show x-labels only on the bottom row of the 2-column grid
        is_bottom = (idx // ncols) == (nrows - 1)
        if is_bottom:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], fontsize=_fs_for_width("tick", fig_width), rotation=12, ha='right')
        else:
            ax.set_xticks(range(len(order)))
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=_fs_for_width("label", fig_width))
        sns.despine(ax=ax)

    for j in range(len(metrics), len(axes_flat)): axes_flat[j].set_visible(False)

    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Plot_Actin_ThreeWay")

def plot_lumen_retention(df, output_dir, _as_figure=False):
    set_paper_style()
    metric = 'A Lumen/Bg'
    if metric not in df.columns: return

    order = [c for c in ['Factin', 'BranchedCortex', 'LinearCortex'] if c in df['Category'].unique()]
    palette = _comp_palette(order)
    plot_df = df[df['Category'].isin(order)].dropna(subset=[metric])

    if plot_df.empty: return

    fig, ax = plt.subplots(figsize=_get_figsize_single_col())
    _violin_box_comp(ax, plot_df, 'Category', metric, order, palette)

    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.9, alpha=0.7)

    try:
        x_pos = {c: i for i, c in enumerate(order)}
        y_range = plot_df[metric].max() - plot_df[metric].min()
        # Matches plot_actin_three_way's step (raised from a smaller value
        # that let stacked brackets' stars collide with the next level).
        step = y_range * 0.22
        global_max = plot_df[metric].max()

        pairs = [('Factin', 'BranchedCortex', 0), ('Factin', 'LinearCortex', 1), ('BranchedCortex', 'LinearCortex', 2)]
        for cond_a, cond_b, level in pairs:
            if cond_a not in x_pos or cond_b not in x_pos: continue
            v_a = plot_df[plot_df['Category'] == cond_a][metric].dropna().values
            v_b = plot_df[plot_df['Category'] == cond_b][metric].dropna().values

            # Anchor to the TRUE max, not a percentile: this metric (A
            # Lumen/Bg) is heavily right-skewed, and a 99th-percentile
            # anchor sat below where the violin's actual tail (drawn out to
            # the true max) reached -- the tail visibly poked through the
            # bracket line. A bracket spanning a skipped-over middle group
            # (e.g. Factin-to-Linear, over Branched) must also clear THAT
            # group's violin, not just the two it's comparing.
            spans_middle = abs(x_pos[cond_a] - x_pos[cond_b]) > 1
            local_y_max = global_max if spans_middle else max(v_a.max(), v_b.max())
            
            stars, label, lw = _build_stat_label(v_a, v_b)
            if stars != "ns":
                _add_bracket(ax, x_pos[cond_a], x_pos[cond_b], local_y_max + (level * step), label, lw=lw,
                             fontsize=_fs("annot"))
    except Exception: pass

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], fontsize=_fs("tick"))
    ax.set_ylabel("Actin Lumen / Background", fontsize=_fs("label"))
    sns.despine()

    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Plot_LumenRetention")

def plot_phenotype_stratified(df, output_dir, _as_figure=False):
    set_paper_style()
    if 'Phenotype_Category' not in df.columns: return

    cont_df = df[(df['Category'].isin(['BranchedCortex', 'LinearCortex'])) & (df['Phenotype_Category'] == 'Continuous')].copy()
    if cont_df.empty: return

    order = [c for c in ['BranchedCortex', 'LinearCortex'] if c in cont_df['Category'].unique()]
    palette = _comp_palette(order)
    metrics = [m for m in ['t_cortex', 'A localization', 'Gini_Index', 'A Lumen/Bg'] if m in cont_df.columns]

    if not metrics: return

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig_width = ncols * 3.8
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows * 4.2))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    metric_labels = METRIC_LABELS

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        m_df = cont_df.dropna(subset=[metric])

        _violin_box_comp(ax, m_df, 'Category', metric, order, palette)
        # These actin metrics are all non-negative — always start the axis at 0
        ax.set_ylim(bottom=0)

        try:
            # Branched vs Linear comparison
            v_b = m_df[m_df['Category'] == 'BranchedCortex'][metric].dropna().values
            v_l = m_df[m_df['Category'] == 'LinearCortex'][metric].dropna().values
            
            local_y_max = max(v_b.max(), v_l.max())
            stars, label, lw = _build_stat_label(v_b, v_l)

            if stars != "ns":
                _add_bracket(ax, 0, 1, local_y_max, label, lw=lw, fontsize=_fs_for_width("annot", fig_width))
        except Exception: pass

        # Show x-labels only on the bottom row of the 2-column grid
        is_bottom = (idx // ncols) == (nrows - 1)
        if is_bottom:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], fontsize=_fs_for_width("tick", fig_width), rotation=12, ha='right')
        else:
            ax.set_xticks(range(len(order)))
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=_fs_for_width("label", fig_width))
        sns.despine(ax=ax)

    for j in range(len(metrics), len(axes_flat)): axes_flat[j].set_visible(False)

    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Plot_Phenotype_Stratified_Continuous")

# ─────────────────────────────────────────────────────────────────────────────
# 13. CORTEX-FORMING-ONLY COMPANION FIGURES
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY THESE FIGURES EXIST
# -----------------------
# In both nucleated conditions (BranchedCortex and LinearCortex), the majority
# of GUVs fail to form a membrane-associated cortex.  They are classified as:
#
#   Empty   — no cortex at all; every cortex metric is zero or near-zero
#   Lumenal — actin ended up inside the GUV, not at the membrane
#
# Including these GUVs in population-level comparisons is like computing the
# average running speed of a group that is mostly spectators who never ran.
# The non-runners all contribute a speed of 0, dragging the group median
# toward zero and hiding real differences between the actual runners.
#
# The companion figures here remove the Empty and Lumenal GUVs first, so
# medians and violin shapes reflect only GUVs that successfully formed a
# membrane-associated actin structure.
#
# WHAT IS FILTERED OUT
# --------------------
#   Phenotype_Category in { Empty, Lumenal, Excluded }
#
# WHAT IS KEPT PER CONDITION
# --------------------------
#   BranchedCortex : Sparse, Patchy, Continuous
#   LinearCortex   : Sparse, Continuous
#   Factin         : Shell  (the membrane-associated shell form)
# ─────────────────────────────────────────────────────────────────────────────

# These phenotype labels represent GUVs with NO membrane-associated cortex.
# Think of them as the "did not form" group — they are excluded from
# cortex-only figures.
#
# IMPORTANT: these must match analysis_phenotype.py's actual return values
# EXACTLY, including case. The classifiers there return "Excluded",
# "Lumenal", "Empty" (proper case) — not the uppercase forms that used to
# be here. Because Python string comparison is case-sensitive, the old
# uppercase set silently matched zero rows: Lumenal and Excluded GUVs were
# being counted as "cortex-forming" in every cortex-only figure and
# statistic, with no error raised. If you ever rename a phenotype in
# analysis_phenotype.py, update this set in the same commit.
_NON_CORTEX_PHENOTYPES = frozenset({'Empty', 'Lumenal', 'Excluded'})


def _filter_cortex_forming(df):
    """
    Returns a copy of df containing only GUVs with a membrane-associated
    actin structure — rows whose Phenotype_Category is NOT in
    { Empty, Lumenal, Excluded }.

    Analogy: imagine your data is a list of petri dishes.  Empty dishes are
    blank (nothing grew), Lumenal dishes grew something but in the wrong
    compartment.  This function keeps only the dishes where the experiment
    actually worked.

    Parameters
    ----------
    df : pandas DataFrame with at least a 'Phenotype_Category' column

    Returns
    -------
    filtered_df : a filtered copy — the original df is never modified.
    If 'Phenotype_Category' is missing, the original df is returned with a
    warning (graceful fallback — the pipeline will not crash).
    """
    if 'Phenotype_Category' not in df.columns:
        # Skip-and-warn pattern: we cannot filter without phenotype labels,
        # but we do not want to crash the pipeline.  Print a clear message
        # and return the data unchanged.
        print("  ! _filter_cortex_forming: 'Phenotype_Category' column not found — "
              "returning unfiltered data.  Run analysis_phenotype first.")
        return df.copy()

    # ~ means "NOT" — keep rows where phenotype is NOT in the exclusion set
    mask     = ~df['Phenotype_Category'].isin(_NON_CORTEX_PHENOTYPES)
    filtered = df[mask].copy()

    # Print a short report so you can see how many GUVs were removed
    n_before  = len(df)
    n_removed = n_before - len(filtered)
    print(f"  [cortex-forming filter]  "
          f"{n_removed:,} / {n_before:,} GUVs removed "
          f"(Empty + Lumenal + Excluded)  →  {len(filtered):,} remaining")

    return filtered


def plot_actin_three_way_cortex_only(df, output_dir, _as_figure=False):
    """
    Cross-condition comparison of Factin / BranchedCortex / LinearCortex
    on actin metrics, restricted to cortex-forming GUVs only.

    This is the direct companion to plot_actin_three_way().
    The layout, statistics, and formatting are identical — the only
    difference is that Empty and Lumenal GUVs are removed before plotting.

    The figure title annotation tells the reader exactly how many GUVs
    were removed so the filtering is fully transparent.

    BIOLOGICAL QUESTION ANSWERED
    ----------------------------
    "Among GUVs that successfully formed a membrane-associated actin
    structure, how do localization, heterogeneity, and cortex thickness
    differ between the three conditions?"
    """
    # ── Count how many actin-condition GUVs exist before the filter ──────────
    # We use this to show "N removed" on the figure, so the reader can judge
    # how large the excluded fraction is.
    actin_conds = ['Factin', 'BranchedCortex', 'LinearCortex']
    n_before    = len(df[df['Category'].isin(actin_conds)])
    plot_df     = _filter_cortex_forming(df)  # <-- the key filtering step
    n_after     = len(plot_df[plot_df['Category'].isin(actin_conds)])
    n_excluded  = n_before - n_after

    set_paper_style()
    order   = [c for c in actin_conds if c in plot_df['Category'].unique()]
    palette = _comp_palette(order)
    metrics = [m for m in ['A localization', 'Gini_Index', 't_cortex']
               if m in plot_df.columns]

    if not metrics:
        # Nothing to plot — skip gracefully rather than crash
        return

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig_width = ncols * 3.8
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_width, nrows * 4.2))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    metric_labels = METRIC_LABELS

    for idx, metric in enumerate(metrics):
        ax   = axes_flat[idx]
        m_df = plot_df.dropna(subset=[metric])

        _violin_box_comp(ax, m_df, 'Category', metric, order, palette)
        # These actin metrics are all non-negative — always start the axis at 0
        ax.set_ylim(bottom=0)

        # ── Significance brackets ─────────────────────────────────────────
        try:
            x_pos   = {c: i for i, c in enumerate(order)}
            y_range = m_df[metric].max() - m_df[metric].min()
            # See plot_actin_three_way for why this was raised from 0.12.
            step    = y_range * 0.20
            global_max = m_df[metric].max()

            pairs = [
                ('Factin',         'BranchedCortex', 0),
                ('Factin',         'LinearCortex',   1),
                ('BranchedCortex', 'LinearCortex',   2),
            ]
            for cond_a, cond_b, level in pairs:
                if cond_a not in x_pos or cond_b not in x_pos:
                    continue
                v_a = m_df[m_df['Category'] == cond_a][metric].dropna().values
                v_b = m_df[m_df['Category'] == cond_b][metric].dropna().values
                # See plot_actin_three_way: a bracket spanning a skipped-over
                # middle group must clear THAT group's violin too.
                spans_middle = abs(x_pos[cond_a] - x_pos[cond_b]) > 1
                y_max = global_max if spans_middle else max(v_a.max(), v_b.max())
                stars, label, lw = _build_stat_label(v_a, v_b)
                if stars != "ns":
                    _add_bracket(ax, x_pos[cond_a], x_pos[cond_b],
                                 y_max + (level * step), label, lw=lw,
                                 fontsize=_fs_for_width("annot", fig_width))
        except Exception:
            pass

        # X-axis labels only on the bottom row (cleaner grid)
        is_bottom = (idx // ncols) == (nrows - 1)
        if is_bottom:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(
                [CONDITION_LABELS.get(c, c) for c in order],
                fontsize=_fs_for_width("tick", fig_width), rotation=12, ha='right')
        else:
            ax.set_xticks(range(len(order)))
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=_fs_for_width("label", fig_width))
        sns.despine(ax=ax)

    # Hide unused panels (e.g. bottom-right when 3 metrics in a 2-col grid)
    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    # ── Italic footnote telling the reader this is a filtered view ────────────
    # "N removed" makes the exclusion fully transparent — no hidden filtering.
    # Footnote is intentionally smaller than the main annotation size.
    fig.text(
        0.5, -0.01,
        f"Cortex-forming GUVs only  ·  Empty + Lumenal excluded  "
        f"({n_excluded:,} GUVs removed from actin conditions)",
        ha='center', va='top', fontsize=_fs_for_width("annot", fig_width) * 0.78, color='#666666', style='italic',
    )

    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Plot_Actin_ThreeWay_CortexOnly")


def plot_lumen_retention_cortex_only(df, output_dir, _as_figure=False):
    """
    A Lumen/Bg ratio comparison restricted to cortex-forming GUVs only.

    Companion to plot_lumen_retention().

    NOTE ON BIOLOGICAL INTERPRETATION SHIFT
    ----------------------------------------
    In the full-population version, Lumenal GUVs are included.  Those GUVs
    have very high A Lumen/Bg (all actin is inside), which inflates the
    condition median.

    Excluding them shifts the question from:
      "What fraction of all GUVs have actin inside?"   (phenotype question)
    to:
      "Among GUVs that formed a membrane cortex, how much actin additionally
       leaked into the lumen?"                         (architecture question)

    Both questions are valid — they answer different things.  The pair of
    figures (full + cortex-only) tells the complete story.
    """
    actin_conds = ['Factin', 'BranchedCortex', 'LinearCortex']
    n_before    = len(df[df['Category'].isin(actin_conds)])
    plot_df     = _filter_cortex_forming(df)
    n_after     = len(plot_df[plot_df['Category'].isin(actin_conds)])
    n_excluded  = n_before - n_after

    set_paper_style()
    metric  = 'A Lumen/Bg'
    if metric not in plot_df.columns:
        return

    order   = [c for c in actin_conds if c in plot_df['Category'].unique()]
    palette = _comp_palette(order)
    m_df    = plot_df[plot_df['Category'].isin(order)].dropna(subset=[metric])

    if m_df.empty:
        return

    fig, ax = plt.subplots(figsize=_get_figsize_single_col())
    _violin_box_comp(ax, m_df, 'Category', metric, order, palette)
    # A Lumen/Bg is a non-negative ratio — always start the axis at 0
    ax.set_ylim(bottom=0)

    # Reference line at 1.0:
    #   below 1.0 → more actin at membrane than inside (good for a cortex)
    #   above 1.0 → more actin inside than at membrane (leaky)
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.9, alpha=0.7)

    # ── Significance brackets ─────────────────────────────────────────────────
    try:
        x_pos      = {c: i for i, c in enumerate(order)}
        y_range    = m_df[metric].max() - m_df[metric].min()
        step       = y_range * 0.22
        global_max = m_df[metric].max()

        pairs = [('Factin', 'BranchedCortex', 0),
                 ('Factin', 'LinearCortex',   1),
                 ('BranchedCortex', 'LinearCortex', 2)]
        for cond_a, cond_b, level in pairs:
            if cond_a not in x_pos or cond_b not in x_pos:
                continue
            v_a = m_df[m_df['Category'] == cond_a][metric].dropna().values
            v_b = m_df[m_df['Category'] == cond_b][metric].dropna().values
            # See plot_lumen_retention: true max (not a percentile), and the
            # global max whenever the bracket spans over a skipped group.
            spans_middle = abs(x_pos[cond_a] - x_pos[cond_b]) > 1
            local_y_max  = global_max if spans_middle else max(v_a.max(), v_b.max())
            stars, label, lw = _build_stat_label(v_a, v_b)
            if stars != "ns":
                _add_bracket(ax, x_pos[cond_a], x_pos[cond_b],
                             local_y_max + (level * step), label, lw=lw,
                             fontsize=_fs("annot"))
    except Exception:
        pass

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order],
                       fontsize=_fs("tick"))
    ax.set_ylabel("Actin Lumen / Background", fontsize=_fs("label"))
    sns.despine()

    plt.tight_layout()
    fig.text(
        0.5, -0.01,
        f"Cortex-forming GUVs only  ·  Empty + Lumenal excluded  "
        f"({n_excluded:,} GUVs removed from actin conditions)",
        ha='center', va='top', fontsize=_fs("annot") * 0.78, color='#666666', style='italic',
    )

    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Plot_LumenRetention_CortexOnly")

def plot_batch_actin_metrics_cortex_only(df, output_dir, _as_figure=False):
    """
    Batch-level violin plots for actin metrics, restricted to cortex-forming
    GUVs only.

    Companion to plot_batch_actin_metrics().

    WHY A BATCH-LEVEL CORTEX-ONLY VIEW MATTERS
    -------------------------------------------
    The overall-median dashed line in the full-population batch figure can
    shift between batches simply because encapsulation efficiency varies — one
    batch may have 20% Empty GUVs, another 80%.  The dashed line then reflects
    "how well did actin encapsulate this batch?" rather than "how good is the
    cortex architecture this batch produced?"

    By filtering to cortex-forming GUVs first, the dashed line and violin
    shapes reflect only the GUVs that actually formed cortex, making
    batch-to-batch comparisons architecturally meaningful.

    N labels on each batch violin tell you how many cortex-forming GUVs
    contributed (after filtering) — so you can spot batches with very few
    surviving GUVs.
    """
    # --- ADJUST FONT SIZES HERE EASILY ---
    FS_TITLE  = 30
    FS_LABEL  = 30
    FS_TICK   = 30
    FS_ANNOT  = 30
    FS_FOOTER = 22

    n_before   = len(df)
    plot_df    = _filter_cortex_forming(df)
    n_excluded = n_before - len(plot_df)

    set_paper_style()
    metrics = ['t_cortex', 'A localization', 'Gini_Index']
    metric_labels = METRIC_LABELS

    actin_conditions = [c for c in CONDITION_ORDER
                        if c in plot_df['Category'].unique() and c != 'Empty']
    if not actin_conditions:
        return  # Nothing to show — skip gracefully

    metrics = [m for m in metrics if m in plot_df.columns]
    if not metrics:
        return

    # Matches the widened panel size in plot_batch_actin_metrics() so both
    # the full-population and cortex-only versions look consistent and
    # both have room for up to ~5 batch columns with N-labels.
    fig_width = len(actin_conditions) * 4.2
    fig, axes = plt.subplots(
        len(metrics), len(actin_conditions),
        figsize=(fig_width, len(metrics) * 4.4),
        squeeze=False,
    )

    for col_idx, condition in enumerate(actin_conditions):
        # Take only cortex-forming GUVs for this condition
        cond_df     = plot_df[plot_df['Category'] == condition].copy()
        cond_df     = _clean_batch_labels(cond_df)  # "260208 #1" -> "#1", sequential per condition
        batch_order = sorted(cond_df['Batch_Label'].unique())
        cond_color  = CONDITION_PALETTE.get(condition, '#6B7B8C')

        for row_idx, metric in enumerate(metrics):
            ax  = axes[row_idx, col_idx]
            sub = cond_df[['Batch_Label', metric]].dropna(subset=[metric]).copy()

            if sub.empty:
                ax.set_visible(False)
                continue

            groups = [sub.loc[sub['Batch_Label'] == b, metric].values
                      for b in batch_order]
            colors = [cond_color] * len(batch_order)

            _draw_standard_violin_box(ax, groups, range(len(batch_order)),
                                      colors, widths=0.12)

            # Dashed line = overall median of CORTEX-FORMING GUVs for this
            # condition.  No longer dragged down by Empty zeros.
            overall_median = sub[metric].median()
            ax.axhline(overall_median, color='#2d7fb8', linestyle='--',
                       linewidth=1.0, alpha=0.65)

            if row_idx == 0:
                ax.set_title(CONDITION_LABELS.get(condition, condition),
                             fontsize=FS_TITLE, fontweight='bold', pad=14)
            if col_idx == 0:
                ax.set_ylabel(metric_labels.get(metric, metric), fontsize=FS_LABEL, fontweight='bold')

            # X-labels only on the bottom row
            if row_idx == len(metrics) - 1:
                ax.set_xticks(range(len(batch_order)))
                ax.set_xticklabels(batch_order, fontsize=FS_TICK,
                                   rotation=30, ha='right', fontweight='bold')
            else:
                ax.set_xticks(range(len(batch_order)))
                ax.tick_params(labelbottom=False)

            # --- EXPLICITLY STYLE Y-TICKS WITH FS_TICK ---
            ax.tick_params(axis='y', labelsize=FS_TICK)
            for label in ax.get_yticklabels():
                label.set_fontsize(FS_TICK)
                label.set_fontweight('bold')

            # N label above each violin: cortex-forming count per batch.
            # Anchored to the TRUE max, not the 99th percentile -- see
            # plot_batch_size_distribution for why that let labels collide
            # with the violin's tapering tip on skewed data.
            y_top   = sub[metric].max()
            y_range = sub[metric].max() - sub[metric].min()
            for x_pos, batch in enumerate(batch_order):
                n_ves = sub[sub['Batch_Label'] == batch][metric].count()
                ax.text(x_pos, y_top + y_range * 0.08,
                        f'N={n_ves}', ha='center', va='bottom', rotation=90,
                        fontsize=FS_ANNOT, color='#555555', fontweight='bold')
            if row_idx == 0:
                # Only the top row sits directly under a title, so only it
                # needs extra headroom to keep the rotated N-label clear.
                # Raised from 0.55 -- see plot_batch_size_distribution for why.
                ax.set_ylim(top=ax.get_ylim()[1] + y_range * 0.80)

            sns.despine(ax=ax)

    # --- RESERVED BOTTOM MARGIN FOR LARGE LABELS & FOOTER ---
    fig.subplots_adjust(bottom=0.18, left=0.12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    fig.text(
        0.5, 0.01,
        f"Cortex-forming GUVs only  ·  Empty + Lumenal excluded  "
        f"({n_excluded:,} GUVs removed total)",
        ha='center', va='bottom', fontsize=FS_FOOTER, color='#666666', style='italic',
    )

    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Batch_Actin_Metrics_CortexOnly")

# ─────────────────────────────────────────────────────────────────────────────
# 14. REPRESENTATIVE RADIAL PROFILE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
#
# Draws the Lumenal / Sparse / Continuous representative actin radial-profile
# comparison for BranchedCortex vs LinearCortex. The data behind this figure
# comes from analysis_batch.compute_representative_radial_profiles(), which
# already did the hard work of normalising and averaging the profiles — this
# function's only job is to draw what it's handed.
# ─────────────────────────────────────────────────────────────────────────────

# Fixed draw order (also the legend order) for the three phenotype curves.
# Kept here, next to PHENOTYPE_PALETTE, since it's a plotting-only concern —
# analysis_batch.py has its own copy of this list for the merge/averaging
# step, but the two lists don't need to be the same object, just the same
# spelling (PHENOTYPE_PALETTE keys are the single source of truth for that).
_REPRESENTATIVE_PHENOTYPE_ORDER = ['Lumenal', 'Sparse', 'Continuous']


def _find_alignment_x(values, radii, phenotype):
    """
    Returns the x-position (Normalized_Radius) that gets shifted to
    x = 0. All curves are aligned on their maximum value (peak) to 
    ensure peaks share a common center.
    """
    peak_idx = values.argmax()
    return radii[peak_idx]

# def plot_representative_radial_profiles(rep_df, output_dir, figsize=None, _as_figure=False):
#     """
#     Draws the representative radial actin-intensity profile for Lumenal,
#     Sparse, and Continuous GUVs, with BranchedCortex and LinearCortex
#     overlaid in ONE shared panel.
#     """
#     set_paper_style()

#     conditions = [c for c in CONDITION_ORDER if c in rep_df['Condition'].unique()]
#     if not conditions:
#         return

#     if figsize is None:
#         figsize = (2.2, 4.4)
#     fig_width, fig_height = figsize
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))

#     VIOLIN_PLOT_LATEX_WIDTH_FRACTION    = 0.54
#     RADIAL_PROFILE_LATEX_WIDTH_FRACTION = 0.22
#     RADIAL_PROFILE_FONT_BOOST = VIOLIN_PLOT_LATEX_WIDTH_FRACTION / RADIAL_PROFILE_LATEX_WIDTH_FRACTION
#     LABEL_EXTRA_BOOST  = 1.15
#     RADIAL_PROFILE_LEGEND_BOOST = 2.1

#     def _fs_rp_label(role):
#         return _fs_for_width(role, fig_width) * RADIAL_PROFILE_FONT_BOOST * LABEL_EXTRA_BOOST

#     def _fs_rp_legend(role):
#         return _fs_for_width(role, fig_width) * RADIAL_PROFILE_LEGEND_BOOST

#     fs_label  = _fs_rp_label("label")
#     fs_tick   = _fs_rp_label("tick")
#     fs_legend = _fs_rp_legend("legend")

#     ax.axvline(0.0, color='#999999', linestyle=':', linewidth=1.0, zorder=1)

#     phenotype_handles = {}
#     all_aligned_min = []
#     all_aligned_max = []

#     for condition in conditions:
#         cond_df    = rep_df[rep_df['Condition'] == condition]
#         linestyle  = CONDITION_LINESTYLE.get(condition, '-')

#         for phenotype in _REPRESENTATIVE_PHENOTYPE_ORDER:
#             curve = cond_df[cond_df['Phenotype'] == phenotype].sort_values('Normalized_Radius')
#             if curve.empty:
#                 continue

#             values = curve['Median_Actin_Normalized'].to_numpy()
#             radii  = curve['Normalized_Radius'].to_numpy()

#             alignment_x = _find_alignment_x(values, radii, phenotype)
#             aligned_x   = curve['Normalized_Radius'] - alignment_x

#             color = PHENOTYPE_PALETTE.get(phenotype, '#aaaaaa')

#             line, = ax.plot(
#                 aligned_x, curve['Median_Actin_Normalized'],
#                 color=color, linestyle=linestyle, linewidth=1.6, zorder=3)

#             all_aligned_min.append(aligned_x.min())
#             all_aligned_max.append(aligned_x.max())

#             if phenotype not in phenotype_handles:
#                 phenotype_handles[phenotype] = line

#     ax.set_xlabel("(r − r$_{peak}$) / R", fontsize=fs_label)
#     ax.set_ylabel("Actin intensity (normalized to max)", fontsize=fs_label)
#     ax.tick_params(labelsize=fs_tick)

#     # Force left boundary negative to clear legend
#     # Cap right boundary to the shortest common curve end to align trace terminations
#     common_x_max = min(all_aligned_max)
#     ax.set_xlim(-1.6, common_x_max)
    
#     ax.set_ylim(0, 1.15)

#     ordered_phenotypes = [p for p in _REPRESENTATIVE_PHENOTYPE_ORDER if p in phenotype_handles]

#     branched_handles = [
#         Line2D([0], [0], color=phenotype_handles[p].get_color(),
#                linestyle=CONDITION_LINESTYLE.get('BranchedCortex', '-'), linewidth=1.6)
#         for p in ordered_phenotypes
#     ]
#     linear_handles = [
#         Line2D([0], [0], color=phenotype_handles[p].get_color(),
#                linestyle=CONDITION_LINESTYLE.get('LinearCortex', '--'), linewidth=1.6)
#         for p in ordered_phenotypes
#     ]

#     leg1 = ax.legend(
#         branched_handles, ordered_phenotypes, title="Branched",
#         loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=fs_legend,
#         title_fontsize=fs_legend, frameon=False, handlelength=1.6,
#         borderaxespad=0.2, labelspacing=0.3)
#     ax.add_artist(leg1)

#     # Shifted Linear legend upward to close the gap
#     ax.legend(
#         linear_handles, ordered_phenotypes, title="Linear",
#         loc='upper left', bbox_to_anchor=(0.0, 0.72), fontsize=fs_legend,
#         title_fontsize=fs_legend, frameon=False, handlelength=1.6,
#         borderaxespad=0.2, labelspacing=0.3)

#     sns.despine(ax=ax)

#     plt.tight_layout()

#     if _as_figure:
#         return fig
#     _save_fig(fig, output_dir, "Plot_Representative_Radial_Profiles")

def plot_representative_radial_profiles(rep_df, output_dir, figsize=None, _as_figure=False):
    """
    Draws representative radial actin-intensity profiles normalized to max (0 to 1)
    and aligned at peak position (x = 0) in two side-by-side panels (Branched vs Linear),
    with a horizontal legend below the subplots.
    """
    set_paper_style()

    conditions = [c for c in CONDITION_ORDER if c in rep_df['Condition'].unique()]
    if not conditions:
        return

    # 1. Taller figure height without increasing width (Width: 6.0, Height: 4.8)
    if figsize is None:
        figsize = (6.0, 4.8)
    fig_width, fig_height = figsize
    
    fig, (ax_branched, ax_linear) = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

    fs_label  = _fs("label")
    fs_tick   = _fs("tick")
    fs_legend = _fs("legend")
    fs_title  = _fs("title")

    # Map conditions to subplots
    cond_ax_map = {}
    for c in conditions:
        if 'branched' in c.lower():
            cond_ax_map[c] = (ax_branched, "Branched")
        elif 'linear' in c.lower():
            cond_ax_map[c] = (ax_linear, "Linear")
        else:
            cond_ax_map[c] = (ax_branched, c)

    phenotype_handles = {}

    for ax in (ax_branched, ax_linear):
        ax.axvline(0.0, color='#b0b0b0', linestyle=':', linewidth=1.0, zorder=1)

    for condition in conditions:
        if condition not in cond_ax_map:
            continue

        ax, title_str = cond_ax_map[condition]
        ax.set_title(title_str, fontsize=fs_title, pad=8, fontweight='bold')

        cond_df = rep_df[rep_df['Condition'] == condition]

        # First pass: calculate aligned x-values for all phenotypes in this condition
        curves_data = {}
        non_lumenal_starts = []

        for phenotype in _REPRESENTATIVE_PHENOTYPE_ORDER:
            curve = cond_df[cond_df['Phenotype'] == phenotype].sort_values('Normalized_Radius')
            if curve.empty:
                continue

            raw_values = curve['Median_Actin_Normalized'].to_numpy()
            radii      = curve['Normalized_Radius'].to_numpy()

            # 0-to-1 Normalization Per Curve
            val_min = np.nanmin(raw_values)
            val_max = np.nanmax(raw_values)
            if val_max > val_min:
                values = (raw_values - val_min) / (val_max - val_min)
            else:
                values = raw_values

            alignment_x = _find_alignment_x(values, radii, phenotype)
            aligned_x   = radii - alignment_x

            curves_data[phenotype] = {
                'aligned_x': aligned_x,
                'values': values
            }

            if phenotype != 'Lumenal':
                non_lumenal_starts.append(np.min(aligned_x))

        # Determine target starting x for Lumenal alignment
        target_start_x = np.median(non_lumenal_starts) if non_lumenal_starts else -0.6

        # Plot processed curves
        all_aligned_min = []
        all_aligned_max = []

        for phenotype in _REPRESENTATIVE_PHENOTYPE_ORDER:
            if phenotype not in curves_data:
                continue

            aligned_x = curves_data[phenotype]['aligned_x']
            values    = curves_data[phenotype]['values']

            # 2. Shift Lumenal curve rightward so it starts where the others do
            if phenotype == 'Lumenal' and len(aligned_x) > 0:
                current_start = np.min(aligned_x)
                shift_offset = target_start_x - current_start
                aligned_x = aligned_x + shift_offset

            color = PHENOTYPE_PALETTE.get(phenotype, '#aaaaaa')

            line, = ax.plot(
                aligned_x, values,
                color=color, linestyle='-', linewidth=1.8, zorder=3)

            all_aligned_min.append(np.min(aligned_x))
            all_aligned_max.append(np.max(aligned_x))

            if phenotype not in phenotype_handles:
                phenotype_handles[phenotype] = line

        if all_aligned_min and all_aligned_max:
            ax.set_xlim(min(all_aligned_min) - 0.08, max(all_aligned_max) + 0.08)

    # Common layout and formatting
    for ax in (ax_branched, ax_linear):
        ax.set_xlabel("(r − r$_{peak}$) / R", fontsize=fs_label)
        ax.tick_params(labelsize=fs_tick)
        ax.set_ylim(-0.02, 1.10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        sns.despine(ax=ax)

    ax_branched.set_ylabel("Actin intensity (normalized to max)", fontsize=fs_label)

    # 3. Side-by-side Legend placed horizontally below both plots
    ordered_phenotypes = [p for p in _REPRESENTATIVE_PHENOTYPE_ORDER if p in phenotype_handles]
    legend_handles = [
        Line2D([0], [0], color=PHENOTYPE_PALETTE.get(p, '#aaaaaa'), linestyle='-', linewidth=1.8)
        for p in ordered_phenotypes
    ]

    fig.legend(
        legend_handles, ordered_phenotypes,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(ordered_phenotypes),
        fontsize=fs_legend,
        frameon=False,
        columnspacing=2.0,
        handlelength=1.5
    )

    # Leave bottom margin for external legend
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Plot_Representative_Radial_Profiles")

# ─────────────────────────────────────────────────────────────────────────────
# 15. REPRESENTATIVE CHANNEL IMAGE GRID
# ─────────────────────────────────────────────────────────────────────────────
#
# Composes the per-vesicle membrane/actin "crop" images (saved by
# skeleton.save_channel_crops via main.py) into one grid: one row per
# (Condition, Phenotype) combination, one column per channel. The actual
# vesicle selected for each row comes from
# analysis_batch.select_representative_vesicles().
# ─────────────────────────────────────────────────────────────────────────────

# Preferred row order within a condition. Whichever of these a condition
# actually has gets shown, in this order; anything else falls back to
# alphabetical so a new/renamed phenotype doesn't just vanish silently.
_REPRESENTATIVE_IMAGE_PHENOTYPE_ORDER = [
    'Empty', 'Lumenal', 'Shell', 'Sparse', 'Patchy', 'Continuous'
]


def plot_representative_channel_images(rep_vesicles_df, root_path, output_dir, _as_figure=False):
    """
    Builds one grid figure of representative membrane/actin crop images —
    one row per (Condition, Phenotype) combination that exists in the data,
    two columns (Membrane, Actin).

    WHERE THE IMAGES COME FROM: this function does NOT touch the original
    microscopy files. It re-loads the small, already-cropped PNGs that
    skeleton.save_channel_crops() saved during the per-vesicle pipeline run
    (one pair per vesicle, living right next to that vesicle's
    Analysis_Results.csv row). The path to each image is rebuilt the same
    way file_handling.py rebuilds Batch_ID/Region_ID:

        root_path / Batch_ID / Region_ID / "Vesicle_<id>_Membrane_Crop.png"
        root_path / Batch_ID / Region_ID / "Vesicle_<id>_Actin_Crop.png"

    If a particular image file can't be found (e.g. main.py was run with
    an older version of skeleton.py, before save_channel_crops existed),
    that panel is left blank with an "image not found" note instead of
    crashing the whole figure.

    Parameters
    ----------
    rep_vesicles_df : pandas DataFrame — output of
                       analysis_batch.select_representative_vesicles(), with
                       columns Category, Phenotype_Category, Batch_ID,
                       Region_ID, Vesicle id, N_in_group, Selection_Metric.
    root_path       : str — the same ROOT_PATH used everywhere else (the
                       top-level Output folder that contains every batch).
    output_dir      : str — folder to save the figure into.
    """
    if rep_vesicles_df is None or rep_vesicles_df.empty:
        return

    set_paper_style()

    conditions = [c for c in CONDITION_ORDER if c in rep_vesicles_df['Category'].unique()]

    # Build the ordered list of rows to draw: (condition, phenotype,
    # membrane_path_or_None, actin_path_or_None, n_in_group).
    rows_to_draw = []
    for condition in conditions:
        cond_rows = rep_vesicles_df[rep_vesicles_df['Category'] == condition]

        present = list(cond_rows['Phenotype_Category'].unique())
        ordered = [p for p in _REPRESENTATIVE_IMAGE_PHENOTYPE_ORDER if p in present]
        ordered += sorted(p for p in present if p not in ordered)

        for phenotype in ordered:
            r = cond_rows[cond_rows['Phenotype_Category'] == phenotype].iloc[0]
            vid    = int(r['Vesicle id'])
            folder = os.path.join(root_path, str(r['Batch_ID']), str(r['Region_ID']))

            membrane_path = os.path.join(folder, f"Vesicle_{vid}_Membrane_Crop.png")
            actin_path    = os.path.join(folder, f"Vesicle_{vid}_Actin_Crop.png")

            rows_to_draw.append((
                condition, phenotype,
                membrane_path if os.path.exists(membrane_path) else None,
                actin_path if os.path.exists(actin_path) else None,
                int(r['N_in_group']),
            ))

    if not rows_to_draw:
        return

    nrows = len(rows_to_draw)
    row_height_in = 1.3
    fig_width_in  = 3.4
    fig, axes = plt.subplots(
        nrows, 2, figsize=(fig_width_in, row_height_in * nrows), squeeze=False)

    label_fontsize = _fs_for_width("label", fig_width_in)
    title_fontsize = _fs_for_width("title", fig_width_in)
    annot_fontsize = _fs_for_width("annot", fig_width_in)

    for row_idx, (condition, phenotype, mpath, apath, n) in enumerate(rows_to_draw):
        ax_m, ax_a = axes[row_idx, 0], axes[row_idx, 1]

        for ax, path in [(ax_m, mpath), (ax_a, apath)]:
            if path is not None:
                ax.imshow(plt.imread(path))
            else:
                ax.text(0.5, 0.5, "image\nnot found", ha='center', va='center',
                        fontsize=annot_fontsize, color='#999999')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Row label (condition + phenotype + N) drawn as free text to the
        # left of the membrane panel — using ax.text instead of
        # set_ylabel(), because turning the axis ticks/spines off above
        # would otherwise hide a real ylabel too.
        row_label = f"{CONDITION_LABELS.get(condition, condition)}\n{phenotype} (N={n})"
        ax_m.text(-0.12, 0.5, row_label, transform=ax_m.transAxes,
                  ha='right', va='center', fontsize=label_fontsize)

        if row_idx == 0:
            ax_m.set_title("Membrane", fontsize=title_fontsize, fontweight='bold')
            ax_a.set_title("Actin", fontsize=title_fontsize, fontweight='bold')

    plt.tight_layout()

    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Representative_Channel_Images")


# ─────────────────────────────────────────────────────────────────────────────
# 10. UNUSED STUBS
# ─────────────────────────────────────────────────────────────────────────────
def plot_cortex_map(cond_df, condition, output_dir): pass
def plot_correlation_heatmap(corr_matrix, output_dir, suffix="", title=None): pass