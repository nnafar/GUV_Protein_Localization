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
import seaborn as sns
from scipy.stats import mannwhitneyu

# ─────────────────────────────────────────────────────────────────────────────
# 0. GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

CONDITION_ORDER = ["Empty", "Factin", "BranchedCortex", "LinearCortex"]

CONDITION_LABELS = {
    "Empty":          "Empty",
    "Factin":         "F-actin",
    "BranchedCortex": "Branched Cortex",
    "LinearCortex":   "Linear Cortex",
}

PHENOTYPE_PALETTE = {
    "EMPTY":      "#D9E4E9",   
    "EXCLUDED":   "#e8e8e8",   
    "LUMENAL":    "#AAB8C2",   
    "SPARSE":     "#6B7B8C",   
    "SHELL":      "#6B7B8C",   
    "PATCHY":     "#42526E",   
    "CONTINUOUS": "#233040",   
}

CONDITION_PALETTE = {
    "Empty":          "#D9E4E9",   
    "Factin":         "#AAB8C2",   
    "BranchedCortex": "#42526E",   
    "LinearCortex":   "#233040",   
}

PLOT_STYLE = {
    "font_family":     "Arial",
    "font_scale":      1.0,
    "fontsize_title":  0,
    "fontsize_label":  9,
    "fontsize_tick":   8,
    "fontsize_legend": 8,
    "fontsize_annot":  8,
    "axes_linewidth":  0.8,
    "tick_linewidth":  0.8,
    "sns_style":       "ticks",
    "sns_context":     "paper",
    "dpi_raster":      1000,
}

def set_paper_style():
    sns.set_theme(
        style      = PLOT_STYLE["sns_style"],
        context    = PLOT_STYLE["sns_context"],
        font_scale = PLOT_STYLE["font_scale"],
    )
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    [PLOT_STYLE["font_family"], "DejaVu Sans"],
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     PLOT_STYLE["axes_linewidth"],
        "xtick.major.width":  PLOT_STYLE["tick_linewidth"],
        "ytick.major.width":  PLOT_STYLE["tick_linewidth"],
        "xtick.labelsize":    PLOT_STYLE["fontsize_tick"],
        "ytick.labelsize":    PLOT_STYLE["fontsize_tick"],
        "axes.titlesize":     PLOT_STYLE["fontsize_title"],
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

    Parameters:
        fig        : the matplotlib Figure object to save
        output_dir : folder path where the file will be written
        stem       : filename without extension (e.g. "Plot_Radius")
                     → saves as "Plot_Radius.pdf"
    """
    path = os.path.join(output_dir, stem + ".pdf")
    fig.savefig(path, bbox_inches="tight")
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

# ─────────────────────────────────────────────────────────────────────────────
# 2. SIZE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_violin_scatter_box(data, x_col, y_col, title, ylabel, output_dir, filename):
    set_paper_style()
    order = [c for c in CONDITION_ORDER if c in data[x_col].unique()]
    colors = [CONDITION_PALETTE[c] for c in order]
    counts = {c: (data[x_col] == c).sum() for c in order}

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = [data.loc[data[x_col] == c, y_col].dropna().values for c in order]
    _draw_standard_violin_box(ax, groups, range(len(order)), colors, widths=0.10)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in order],
        fontsize=11,
    )
    ax.set_ylabel(ylabel, fontsize=12)
    #ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(-0.6, len(order) - 0.4)

    plt.tight_layout()
    stem = os.path.splitext(filename)[0]  # Strip extension so _save_fig adds .pdf
    _save_fig(fig, output_dir, stem)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PHENOTYPE COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def plot_phenotype_composition(df, x_col, phenotype_col, output_dir, filename, _as_figure=False):
    set_paper_style()
    order = [c for c in CONDITION_ORDER if c in df[x_col].unique()]
    pheno_order = ["EMPTY", "LUMENAL", "SHELL", "SPARSE", "PATCHY", "CONTINUOUS", "EXCLUDED"]

    counts = df.groupby([x_col, phenotype_col]).size().unstack(fill_value=0).reindex(order)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pheno_present = [p for p in pheno_order if p in pct.columns]
    pct = pct[pheno_present]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(order))
    
    for pheno in pheno_present:
        vals = pct[pheno].values
        ax.bar(
            range(len(order)), vals, bottom=bottom, width=0.7,
            color=PHENOTYPE_PALETTE.get(pheno, "#aaaaaa"),
            edgecolor="#333333", linewidth=0.8, label=pheno
        )
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 8:  # only label segments wide enough to read
                ax.text(j, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold",
                        path_effects=[pe.Stroke(linewidth=2.5, foreground='black'), pe.Normal()])
        bottom += vals

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in order], fontsize=11)
    ax.set_ylabel("Percentage of GUVs (%)", fontsize=11)
    ax.set_ylim(0, 108)

    handles = [mpatches.Patch(facecolor=PHENOTYPE_PALETTE.get(p, "#aaa"), edgecolor="#333333", linewidth=0.8, label=p) for p in pheno_present]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, frameon=False, title="Phenotype")

    plt.tight_layout()
    if _as_figure:
        return fig  # Return to figure_assembly.py without saving
    stem = os.path.splitext(filename)[0]
    _save_fig(fig, output_dir, stem)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PAIR PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_pairplot(cond_df, output_dir, filename_suffix="", _as_figure=False):
    from scipy.stats import spearmanr
    set_paper_style()

    # RADIUS INJECTED AS A PARAMETER HERE
    metrics = ["Refined Radius (um)", "t_cortex", "A localization", "Gini_Index"]
    cols_present = [c for c in metrics if c in cond_df.columns]
    plot_df = cond_df[cols_present + ["Phenotype_Category"]].dropna(subset=cols_present)
    plot_df = plot_df[plot_df["Phenotype_Category"] != "EMPTY"]

    if plot_df.empty:
        return

    phenotypes = plot_df["Phenotype_Category"].unique().tolist()
    pheno_order_all = ["SHELL", "SPARSE", "PATCHY", "CONTINUOUS", "EXCLUDED"]
    phenotypes_sorted = [p for p in pheno_order_all if p in phenotypes]
    palette = {p: PHENOTYPE_PALETTE.get(p, "#aaaaaa") for p in phenotypes_sorted}

    PAIRPLOT_RENAME = {
        "Refined Radius (um)": "GUV Radius (µm)",
        "t_cortex": "Cortex Thickness (µm)",
        "A localization": "Localization Score",
        "Gini_Index": "Gini Index",
    }
    
    plot_df = plot_df.rename(columns=PAIRPLOT_RENAME)
    cols_display = [PAIRPLOT_RENAME.get(c, c) for c in cols_present]

    def _corr_upper_closure(x, y, **kwargs):
        ax = plt.gca()
        if getattr(ax, '_corr_drawn', False):
            return
        ax._corr_drawn = True

        ax.set_xticks([])
        ax.set_yticks([])

        col_x = x.name
        col_y = y.name
        full_x = plot_df[col_x]
        full_y = plot_df[col_y]

        mask = ~(np.isnan(full_x.values) | np.isnan(full_y.values))
        valid_x = full_x.values[mask]
        valid_y = full_y.values[mask]

        if len(valid_x) < 5 or np.std(valid_x) == 0 or np.std(valid_y) == 0:
            ax.set_facecolor("#e5e5e5")
            ax.text(0.5, 0.5, "n.d.", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#999999")
            return

        r, p = spearmanr(valid_x, valid_y)
        color = plt.cm.RdBu_r((r + 1) / 2)
        ax.set_facecolor(color)

        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = "white" if luminance < 0.55 else "#222222"
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        ax.text(0.5, 0.5, f"r={r:.2f}\n{stars}", transform=ax.transAxes, ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        
        g = sns.PairGrid(
            plot_df, vars=cols_display, hue="Phenotype_Category",
            palette=palette, hue_order=phenotypes_sorted,
            diag_sharey=False, height=2.4, aspect=1.0  # Slightly reduced scale to support the 4x4 layout
        )
        
        g.map_diag(sns.kdeplot, fill=True, alpha=0.50, linewidth=1.2)
        g.map_lower(sns.scatterplot, s=12, alpha=0.45, edgecolor="none", linewidth=0)
        g.map_upper(_corr_upper_closure)

    n_vars = len(cols_display)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            ax = g.axes[i, j]
            for spine in ax.spines.values():
                spine.set_visible(False)

    g.figure.subplots_adjust(bottom=0.10)
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cax = g.figure.add_axes([0.20, 0.025, 0.55, 0.022])
    cbar = g.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Spearman ρ", fontsize=PLOT_STYLE["fontsize_annot"])
    cbar.ax.tick_params(labelsize=PLOT_STYLE["fontsize_annot"] - 2)

    g.add_legend(
        title          = "Phenotype",
        fontsize       = 11,
        title_fontsize = 12,
        bbox_to_anchor = (1.0, 0.5),
        loc            = "center left",
        markerscale    = 2.5,
    )
    
    condition_label = filename_suffix.lstrip("_")
    g.figure.suptitle(f"Correlation Matrix & Pair Plot — {CONDITION_LABELS.get(condition_label, condition_label)}", y=1.02, fontsize=12)

    stem = f"Correlation_Matrix_PairPlot{filename_suffix}"
    if _as_figure:
        return g.figure  # Return to figure_assembly.py without saving
    g.figure.savefig(os.path.join(output_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(g.figure)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PHENOTYPE CHARACTERISTICS MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def _add_sig_brackets(ax, df, metric, pheno_order):
    """
    Draws significance brackets between adjacent phenotype groups in the
    Phenotype Characteristics Matrix.

    Stars above the bracket = statistical significance (p-value).
    Bracket line thickness  = practical effect size (Cliff's Delta).
    No extra text is added — all information lives in the visual.
    """
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
                group_top = max(np.nanpercentile(g1, 95), np.nanpercentile(g2, 95))
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

    current_top = ax.get_ylim()[1]
    base_y = max(current_top, max(s[4] for s in sig)) + step * 0.3

    for i, (x1, x2, stars, lw, _) in enumerate(sig):
        y      = base_y + step * i
        tick_h = step * 0.15
        # Single flat horizontal bar — no vertical tick lines
        ax.plot([x1, x2], [y, y], color="black", lw=lw)
        # Stars annotation — just the stars, no delta text
        ax.text((x1 + x2) / 2, y + step * 0.05, stars,
                ha="center", va="bottom", fontsize=8.5, color="black")

    ax.set_ylim(ax.get_ylim()[0], base_y + step * (len(sig) + 0.8))

def generate_category_panel(df, output_dir, _as_figure=False):
    set_paper_style()
    metrics = ["t_cortex", "A localization", "Gini_Index"]
    metric_lbls = {"t_cortex": "Cortex Thickness (µm)", "A localization": "Localization Score", "Gini_Index": "Gini Index"}
    cond_order = [c for c in CONDITION_ORDER if c in df["Category"].unique() and c != "Empty"]
    
    pheno_all = ["SHELL", "SPARSE", "PATCHY", "CONTINUOUS"]
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
    fig_width = 1.35 * total_cats + 0.8 * len(cond_order) # Explicit aspect scale map
    
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
    # conditions, then add 15% breathing room for labels and brackets.
    # _add_sig_brackets can still push the limit higher if needed — this just
    # gives it a tighter starting point to work from.
    relevant_df = df[df["Category"].isin(cond_order)]
    y_caps = {}
    for metric in metrics:
        vals = relevant_df[metric].dropna()
        if len(vals) == 0:
            y_caps[metric] = None
        else:
            p99 = np.nanpercentile(vals, 99)
            y_caps[metric] = p99 * 1.15  # 15% headroom above the 99th percentile

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
            
            _add_sig_brackets(ax, sub, metric, pheno_present)

            ax.set_xticks(positions)

            # Only show x-axis labels on the bottom row.
            # For all other rows, tick marks stay (for visual alignment) but
            # the text labels are hidden — no need to repeat them every row.
            if ri == len(metrics) - 1:
                ax.set_xticklabels(pheno_present, fontsize=PLOT_STYLE["fontsize_tick"], rotation=20)
            else:
                ax.tick_params(labelbottom=False)  # Hide text, keep tick marks

            # Show the condition name as a column title on the top row only.
            # Now that x-labels only appear once at the bottom, the reader needs
            # the column header to know which condition each column represents.
            if ri == 0:
                ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=11, fontweight='bold', pad=10)

            if ci == 0:
                ax.set_ylabel(metric_lbls.get(metric, metric), fontsize=PLOT_STYLE["fontsize_label"])
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5), squeeze=False)
    ax_flat = axes.flatten()

    for idx, condition in enumerate(conditions):
        ax = ax_flat[idx]
        cond_df = df[df['Category'] == condition].copy()
        batch_order = sorted(cond_df['Batch_Label'].unique())
        cond_color = CONDITION_PALETTE.get(condition, '#6B7B8C')
        
        sub = cond_df[['Batch_Label', metric]].dropna(subset=[metric])
        if sub.empty:
            ax.set_visible(False)
            continue

        groups = [sub.loc[sub['Batch_Label'] == b, metric].values for b in batch_order]
        colors = [cond_color] * len(batch_order)
        
        _draw_standard_violin_box(ax, groups, range(len(batch_order)), colors, widths=0.12)

        overall_median = sub[metric].median()
        ax.axhline(overall_median, color='#2d7fb8', linestyle='--', linewidth=1.1, alpha=0.7)

        n_per_batch = [sub[sub['Batch_Label'] == b][metric].count()
                       for b in batch_order]

        ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=11, fontweight='bold')
        ax.set_ylabel('GUV Radius (µm)', fontsize=10)
        is_bottom_row = (idx // ncols) == (nrows - 1) or (idx + ncols) >= len(conditions)
        if is_bottom_row:
            ax.set_xticks(range(len(batch_order)))
            ax.set_xticklabels(
                [f'{b}\nN={n:,}' for b, n in zip(batch_order, n_per_batch)],
                fontsize=9, rotation=30, ha='right')
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
    set_paper_style()
    if 'Phenotype_Category' not in df.columns: return

    conditions = [c for c in CONDITION_ORDER if c in df['Category'].unique() and c != 'Empty']
    if not conditions: return

    pheno_order_all = ["EMPTY", "LUMENAL", "SHELL", "SPARSE", "PATCHY", "CONTINUOUS", "EXCLUDED"]
    ncols = min(2, len(conditions))
    nrows = int(np.ceil(len(conditions) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, nrows * 4.5), squeeze=False)
    ax_flat = axes.flatten()

    for idx, condition in enumerate(conditions):
        ax = ax_flat[idx]
        cond_df = df[df['Category'] == condition].copy()
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
                            fontsize=7.5, color='white', fontweight='bold',
                            path_effects=[pe.Stroke(linewidth=0.5, foreground='black'), pe.Normal()])
            bottom += vals

        for x_pos, batch in enumerate(batch_order):
            n_ves = len(cond_df[cond_df['Batch_Label'] == batch])
            ax.text(x_pos, 102, f'N={n_ves}', ha='center', va='bottom', fontsize=7.5, color='#555555')

        ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=11, fontweight='bold')
        ax.set_ylabel('GUVs (%)', fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_xticks(range(len(batch_order)))
        ax.set_xticklabels(batch_order, fontsize=9, rotation=30, ha='right')

    all_phenos_shown = [p for p in pheno_order_all if p in df['Phenotype_Category'].unique() and p != 'EMPTY']
    legend_patches = [mpatches.Patch(facecolor=PHENOTYPE_PALETTE.get(p, '#aaa'), edgecolor='#333333', linewidth=0.8, label=p) for p in all_phenos_shown]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(legend_patches), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.05), title='Phenotype')

    for idx in range(len(conditions), len(ax_flat)): ax_flat[idx].set_visible(False)
    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Batch_Phenotype_Composition_All_Conditions")

# ─────────────────────────────────────────────────────────────────────────────
# 8. BATCH ACTIN METRICS
# ─────────────────────────────────────────────────────────────────────────────

def plot_batch_actin_metrics(df, output_dir, _as_figure=False):
    set_paper_style()
    metrics = ['t_cortex', 'A localization', 'Gini_Index', 'Solidity']
    metric_labels = {
        't_cortex':       'Cortex Thickness (µm)',
        'A localization': 'Localization Score',
        'Gini_Index':     'Gini Index',
        'Solidity':       'Solidity',
    }
    
    actin_conditions = [c for c in CONDITION_ORDER if c in df['Category'].unique() and c != 'Empty']
    if not actin_conditions: return

    metrics = [m for m in metrics if m in df.columns]
    if not metrics: return

    fig, axes = plt.subplots(len(metrics), len(actin_conditions), figsize=(len(actin_conditions) * 4.5, len(metrics) * 3.2), squeeze=False)

    for col_idx, condition in enumerate(actin_conditions):
        cond_df = df[df['Category'] == condition].copy()
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

            overall_median = sub[metric].median()
            ax.axhline(overall_median, color='#2d7fb8', linestyle='--', linewidth=1.0, alpha=0.65)

            if row_idx == 0: ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=11, fontweight='bold', pad=6)
            if col_idx == 0: ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9)

            # Bottom row gets batch labels + N count on a second line.
            # N is shown once (bottom row) — it's the same across all metric rows.
            if row_idx == len(metrics) - 1:
                n_per_batch = [sub[sub['Batch_Label'] == b][metric].count()
                               for b in batch_order]
                ax.set_xticks(range(len(batch_order)))
                ax.set_xticklabels(
                    [f'{b}\nN={n:,}' for b, n in zip(batch_order, n_per_batch)],
                    fontsize=8, rotation=30, ha='right')
            else:
                ax.set_xticks(range(len(batch_order)))
                ax.tick_params(labelbottom=False)

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

    Mapping (Romano et al. thresholds):
        |δ| < 0.147  → 0.8 px  (negligible — barely a whisper)
        |δ| < 0.330  → 1.4 px  (small)
        |δ| < 0.474  → 2.2 px  (medium)
        |δ| ≥ 0.474  → 3.2 px  (large — unmistakable)
    """
    abs_d = abs(delta)
    if abs_d < 0.147:
        return 0.8
    elif abs_d < 0.330:
        return 1.4
    elif abs_d < 0.474:
        return 2.2
    else:
        return 3.2

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
        return "ns", "ns", 0.5  # Draw thin line so every comparison is visible

    delta = _cliffs_delta(vals_a, vals_b)
    lw    = _delta_linewidth(delta)  # Thick line = large effect, thin = small

    return stars, stars, lw  # label is just the stars — clean and uncluttered

def _add_bracket(ax, x1, x2, y_top, label, lw=0.9, color='black'):
    """
    Draws a significance bracket between two groups on the plot.

    The stars annotation tells you *if* the difference is significant (p-value).
    The line thickness tells you *how big* the difference is (Cliff's Delta).
    No extra text is added — all the information lives in the visual itself.

    Parameters:
        ax    : the matplotlib Axes object to draw on
        x1, x2: the x-positions of the two groups being compared
        y_top : the data y-value just above the tallest bar (bracket root)
        label : the star string, e.g. "***"
        lw    : line width encoding effect size (from _delta_linewidth)
        color : line and text colour (default black)
    """
    # y_top is the exact position of the bracket line.
    # The text floats just above it, scaled to the current y-axis range.
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_text  = y_top + y_range * 0.03

    # Horizontal bar at exactly y_top
    ax.plot([x1, x2], [y_top, y_top], lw=lw, color=color)

    # Star / ns annotation just above the line
    ax.text(
        (x1 + x2) / 2, y_text, label,
        ha='center', va='bottom',
        fontsize=8, color=color,
    )

    # Expand y-axis so text is never clipped
    needed_top = y_text + y_range * 0.06
    if needed_top > ax.get_ylim()[1]:
        ax.set_ylim(ax.get_ylim()[0], needed_top)


def _draw_pairwise_brackets(ax, m_df, metric, order, pairs):
    """
    Draws significance brackets for every pair in `pairs`, stacked cleanly
    above the 99th-percentile of the data.

    WHY THIS HELPER EXISTS
    ----------------------
    The old approach computed a local y_max per pair (max of the two groups)
    then added level*step on top.  When pairs have different local maxima,
    brackets can end up at the same height or even reverse order — they overlap.

    This helper uses a SINGLE global y_base (p99 of all data) so every bracket
    starts from the same reference point and steps upward by a fixed amount.
    All comparisons are always shown — "ns" gets a thin grey line so the reader
    can see that the test was run and found nothing.

    Parameters
    ----------
    ax     : the Axes to draw on
    m_df   : DataFrame already filtered to the relevant conditions + metric
    metric : column name string
    order  : list of condition names in x-axis order
    pairs  : list of (cond_a, cond_b) tuples — one bracket per pair
    """
    x_pos = {c: i for i, c in enumerate(order)}
    vals  = m_df[metric].dropna()
    if len(vals) < 2:
        return

    # Bracket base = 99th percentile (ignores extreme outliers that would
    # push the first bracket far above the violin body)
    y_base = float(vals.quantile(0.99))
    y_span = float(vals.quantile(0.99) - vals.quantile(0.01))
    # Step is 20 % of the data span, minimum 5 % of y_base so brackets
    # are always visible even when the span is very small
    step = max(y_span * 0.20, abs(y_base) * 0.05, 1e-6)

    level = 0
    for cond_a, cond_b in pairs:
        if cond_a not in x_pos or cond_b not in x_pos:
            continue
        v_a = m_df[m_df['Category'] == cond_a][metric].dropna().values
        v_b = m_df[m_df['Category'] == cond_b][metric].dropna().values
        if len(v_a) < 5 or len(v_b) < 5:
            continue
        stars, label, lw = _build_stat_label(v_a, v_b)
        _add_bracket(ax, x_pos[cond_a], x_pos[cond_b],
                     y_base + level * step, label, lw=lw)
        level += 1

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
        rotation=rotation, ha='right', fontsize=9,
    )

def plot_radius_all_conditions(df, output_dir):
    set_paper_style()
    metric = 'Refined Radius (um)'
    if metric not in df.columns: return

    order = [c for c in CONDITION_ORDER if c in df['Category'].unique()]
    palette = _comp_palette(order)
    plot_df = df[df['Category'].isin(order)].dropna(subset=[metric])

    fig, ax = plt.subplots(figsize=(7, 5))
    _violin_box_comp(ax, plot_df, 'Category', metric, order, palette)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("GUV Radius (µm)", fontsize=11)
    #ax.set_title("GUV Radius Across Conditions", fontsize=12, pad=10)
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

    fig, ax = plt.subplots(figsize=(7, 5))
    _violin_box_comp(ax, plot_df, 'Category', metric, order, palette)
 
    _fmt_xtick(ax, order)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("GUV Radius (µm)", fontsize=11)
    #ax.set_title("GUV Radius: Cortex vs Empty", fontsize=12, pad=10)
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
    metrics = [m for m in ['A localization', 'Gini_Index', 't_cortex', 'Solidity']
               if m in plot_df.columns]

    if not metrics: return

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3.8, nrows * 4.2))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    metric_labels = {
        't_cortex':       'Cortex Thickness (µm)',
        'A localization': 'Localization Score',
        'Gini_Index':     'Gini Index',
        'Solidity':       'Solidity',
    }

    _pairs = [('Factin', 'BranchedCortex'), ('Factin', 'LinearCortex'),
              ('BranchedCortex', 'LinearCortex')]

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        m_df = plot_df.dropna(subset=[metric])

        _violin_box_comp(ax, m_df, 'Category', metric, order, palette)

        try:
            _draw_pairwise_brackets(ax, m_df, metric, order, _pairs)
        except Exception:
            pass

        # Show x-labels only on the bottom row of the 2-column grid
        is_bottom = (idx // ncols) == (nrows - 1)
        if is_bottom:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], fontsize=9, rotation=12, ha='right')
        else:
            ax.set_xticks(range(len(order)))
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9)
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

    fig, ax = plt.subplots(figsize=(6, 5))
    _violin_box_comp(ax, plot_df, 'Category', metric, order, palette)

    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.9, alpha=0.7)

    try:
        _pairs = [('Factin', 'BranchedCortex'), ('Factin', 'LinearCortex'),
                  ('BranchedCortex', 'LinearCortex')]
        _draw_pairwise_brackets(ax, plot_df, metric, order, _pairs)
    except Exception: pass

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], fontsize=10)
    ax.set_ylabel("Actin Lumen / Background", fontsize=11)
    sns.despine()

    plt.tight_layout()
    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Plot_LumenRetention")

def plot_phenotype_stratified(df, output_dir, _as_figure=False):
    set_paper_style()
    if 'Phenotype_Category' not in df.columns: return

    cont_df = df[(df['Category'].isin(['BranchedCortex', 'LinearCortex'])) & (df['Phenotype_Category'] == 'CONTINUOUS')].copy()
    if cont_df.empty: return

    order = [c for c in ['BranchedCortex', 'LinearCortex'] if c in cont_df['Category'].unique()]
    palette = _comp_palette(order)
    metrics = [m for m in ['t_cortex', 'A localization', 'Gini_Index', 'A Lumen/Bg'] if m in cont_df.columns]

    if not metrics: return

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3.8, nrows * 4.2))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    metric_labels = {'t_cortex': 'Cortex Thickness (µm)', 'A localization': 'Localization Score', 'Gini_Index': 'Gini Index', 'A Lumen/Bg': 'Actin Lumen / Background'}

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        m_df = cont_df.dropna(subset=[metric])

        _violin_box_comp(ax, m_df, 'Category', metric, order, palette)

        try:
            # Branched vs Linear comparison
            v_b = m_df[m_df['Category'] == 'BranchedCortex'][metric].dropna().values
            v_l = m_df[m_df['Category'] == 'LinearCortex'][metric].dropna().values
            
            y_base = m_df[metric].quantile(0.99)
            stars, label, lw = _build_stat_label(v_b, v_l)
            _add_bracket(ax, 0, 1, y_base, label, lw=lw)
        except Exception: pass

        # Show x-labels only on the bottom row of the 2-column grid
        is_bottom = (idx // ncols) == (nrows - 1)
        if is_bottom:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order], fontsize=9, rotation=12, ha='right')
        else:
            ax.set_xticks(range(len(order)))
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9)
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
#   EMPTY   — no cortex at all; every cortex metric is zero or near-zero
#   LUMENAL — actin ended up inside the GUV, not at the membrane
#
# Including these GUVs in population-level comparisons is like computing the
# average running speed of a group that is mostly spectators who never ran.
# The non-runners all contribute a speed of 0, dragging the group median
# toward zero and hiding real differences between the actual runners.
#
# The companion figures here remove the EMPTY and LUMENAL GUVs first, so
# medians and violin shapes reflect only GUVs that successfully formed a
# membrane-associated actin structure.
#
# WHAT IS FILTERED OUT
# --------------------
#   Phenotype_Category in { EMPTY, LUMENAL, EXCLUDED }
#
# WHAT IS KEPT PER CONDITION
# --------------------------
#   BranchedCortex : SPARSE, PATCHY, CONTINUOUS
#   LinearCortex   : SPARSE, CONTINUOUS
#   Factin         : SHELL  (the membrane-associated shell form)
# ─────────────────────────────────────────────────────────────────────────────

# These phenotype labels represent GUVs with NO membrane-associated cortex.
# Think of them as the "did not form" group — they are excluded from
# cortex-only figures.
_NON_CORTEX_PHENOTYPES = frozenset({'EMPTY', 'LUMENAL', 'EXCLUDED'})


def _filter_cortex_forming(df):
    """
    Returns a copy of df containing only GUVs with a membrane-associated
    actin structure — rows whose Phenotype_Category is NOT in
    { EMPTY, LUMENAL, EXCLUDED }.

    Analogy: imagine your data is a list of petri dishes.  EMPTY dishes are
    blank (nothing grew), LUMENAL dishes grew something but in the wrong
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
          f"(EMPTY + LUMENAL + EXCLUDED)  →  {len(filtered):,} remaining")

    return filtered


def plot_actin_three_way_cortex_only(df, output_dir, _as_figure=False):
    """
    Cross-condition comparison of Factin / BranchedCortex / LinearCortex
    on actin metrics, restricted to cortex-forming GUVs only.

    This is the direct companion to plot_actin_three_way().
    The layout, statistics, and formatting are identical — the only
    difference is that EMPTY and LUMENAL GUVs are removed before plotting.

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
    metrics = [m for m in ['A localization', 'Gini_Index', 't_cortex', 'Solidity']
               if m in plot_df.columns]

    if not metrics:
        # Nothing to plot — skip gracefully rather than crash
        return

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols * 3.8, nrows * 4.2))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    metric_labels = {
        't_cortex':       'Cortex Thickness (µm)',
        'A localization': 'Localization Score',
        'Gini_Index':     'Gini Index',
        'Solidity':       'Solidity',
    }
    _pairs = [('Factin', 'BranchedCortex'), ('Factin', 'LinearCortex'),
              ('BranchedCortex', 'LinearCortex')]

    for idx, metric in enumerate(metrics):
        ax   = axes_flat[idx]
        m_df = plot_df.dropna(subset=[metric])

        _violin_box_comp(ax, m_df, 'Category', metric, order, palette)

        try:
            _draw_pairwise_brackets(ax, m_df, metric, order, _pairs)
        except Exception:
            pass
        # X-axis labels only on the bottom row (cleaner grid)
        is_bottom = (idx // ncols) == (nrows - 1)
        if is_bottom:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(
                [CONDITION_LABELS.get(c, c) for c in order],
                fontsize=9, rotation=12, ha='right')
        else:
            ax.set_xticks(range(len(order)))
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9)
        sns.despine(ax=ax)

    # Hide unused panels (e.g. bottom-right when 3 metrics in a 2-col grid)
    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    # ── Italic footnote telling the reader this is a filtered view ────────────
    # "N removed" makes the exclusion fully transparent — no hidden filtering.
    fig.text(
        0.5, -0.01,
        f"Cortex-forming GUVs only  ·  EMPTY + LUMENAL excluded  "
        f"({n_excluded:,} GUVs removed from actin conditions)",
        ha='center', va='top', fontsize=7, color='#666666', style='italic',
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
    In the full-population version, LUMENAL GUVs are included.  Those GUVs
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

    fig, ax = plt.subplots(figsize=(6, 5))
    _violin_box_comp(ax, m_df, 'Category', metric, order, palette)

    # Reference line at 1.0:
    #   below 1.0 → more actin at membrane than inside (good for a cortex)
    #   above 1.0 → more actin inside than at membrane (leaky)
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.9, alpha=0.7)

    # ── Significance brackets ─────────────────────────────────────────────────
    try:
        _pairs = [('Factin', 'BranchedCortex'), ('Factin', 'LinearCortex'),
                  ('BranchedCortex', 'LinearCortex')]
        _draw_pairwise_brackets(ax, m_df, metric, order, _pairs)
    except Exception:
        pass

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order],
                       fontsize=10)
    ax.set_ylabel("Actin Lumen / Background", fontsize=11)
    sns.despine()

    plt.tight_layout()
    fig.text(
        0.5, -0.01,
        f"Cortex-forming GUVs only  ·  EMPTY + LUMENAL excluded  "
        f"({n_excluded:,} GUVs removed from actin conditions)",
        ha='center', va='top', fontsize=7, color='#666666', style='italic',
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
    batch may have 20% EMPTY GUVs, another 80%.  The dashed line then reflects
    "how well did actin encapsulate this batch?" rather than "how good is the
    cortex architecture this batch produced?"

    By filtering to cortex-forming GUVs first, the dashed line and violin
    shapes reflect only the GUVs that actually formed cortex, making
    batch-to-batch comparisons architecturally meaningful.

    N labels on each batch violin tell you how many cortex-forming GUVs
    contributed (after filtering) — so you can spot batches with very few
    surviving GUVs.
    """
    n_before   = len(df)
    plot_df    = _filter_cortex_forming(df)
    n_excluded = n_before - len(plot_df)

    set_paper_style()
    metrics = ['t_cortex', 'A localization', 'Gini_Index', 'Solidity']
    metric_labels = {
        't_cortex':       'Cortex Thickness (µm)',
        'A localization': 'Localization Score',
        'Gini_Index':     'Gini Index',
        'Solidity':       'Solidity',
    }

    actin_conditions = [c for c in CONDITION_ORDER
                        if c in plot_df['Category'].unique() and c != 'Empty']
    if not actin_conditions:
        return  # Nothing to show — skip gracefully

    metrics = [m for m in metrics if m in plot_df.columns]
    if not metrics:
        return

    fig, axes = plt.subplots(
        len(metrics), len(actin_conditions),
        figsize=(len(actin_conditions) * 4.5, len(metrics) * 3.2),
        squeeze=False,
    )

    for col_idx, condition in enumerate(actin_conditions):
        # Take only cortex-forming GUVs for this condition
        cond_df     = plot_df[plot_df['Category'] == condition].copy()
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
            # condition.  No longer dragged down by EMPTY zeros.
            overall_median = sub[metric].median()
            ax.axhline(overall_median, color='#2d7fb8', linestyle='--',
                       linewidth=1.0, alpha=0.65)

            if row_idx == 0:
                ax.set_title(CONDITION_LABELS.get(condition, condition),
                             fontsize=11, fontweight='bold', pad=6)
            if col_idx == 0:
                ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9)

            # Bottom row: batch labels + N on a second line
            if row_idx == len(metrics) - 1:
                n_per_batch = [sub[sub['Batch_Label'] == b][metric].count()
                               for b in batch_order]
                ax.set_xticks(range(len(batch_order)))
                ax.set_xticklabels(
                    [f'{b}\nN={n:,}' for b, n in zip(batch_order, n_per_batch)],
                    fontsize=8, rotation=30, ha='right')
            else:
                ax.set_xticks(range(len(batch_order)))
                ax.tick_params(labelbottom=False)

            sns.despine(ax=ax)

    plt.tight_layout()
    fig.text(
        0.5, -0.01,
        f"Cortex-forming GUVs only  ·  EMPTY + LUMENAL excluded  "
        f"({n_excluded:,} GUVs removed total)",
        ha='center', va='top', fontsize=7, color='#666666', style='italic',
    )

    if _as_figure:
        return fig
    _save_fig(fig, output_dir, "Batch_Actin_Metrics_CortexOnly")


# ─────────────────────────────────────────────────────────────────────────────
# 10. UNUSED STUBS
# ─────────────────────────────────────────────────────────────────────────────
def plot_cortex_map(cond_df, condition, output_dir): pass
def plot_correlation_heatmap(corr_matrix, output_dir, suffix="", title=None): pass