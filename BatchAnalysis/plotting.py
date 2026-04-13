# -*- coding: utf-8 -*-
"""
PLOTTING MODULE  — improved visuals
------------------------------------
Changes from previous version:
  1. Condition order:  Empty → Factin → BranchedCortex → LinearCortex
     (controls first, then experimental)
  2. Violin plot: box/whisker drawn on top of (not underneath) the scatter,
     with a white fill so the median / IQR lines are always visible.
  3. Colour scheme: a perceptually-uniform blue palette with enough contrast
     to distinguish all phenotypes and conditions.
  4. Spearman heatmap: larger figure, rotated labels, only lower-triangle
     annotations, smaller font — fully legible at publication size.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu


# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# Canonical order for conditions  (controls → experimental)
CONDITION_ORDER = ["Empty", "Factin", "BranchedCortex", "LinearCortex"]

# Human-readable labels for the x-axis
CONDITION_LABELS = {
    "Empty":          "Empty",
    "Factin":         "F-actin",
    "BranchedCortex": "Branched Cortex",
    "LinearCortex":   "Linear Cortex",
}

# ── Phenotype colour map  ────────────────────────────────────────────────────
# Five distinct, blue-family hues that remain separable in greyscale and
# for the most common forms of colour-vision deficiency.
PHENOTYPE_PALETTE = {
    "EMPTY":      "#cfe2f3",   # very light blue
    "EXCLUDED":   "#e8e8e8",   # neutral grey
    "LUMENAL":    "#9ec8e8",   # soft cornflower
    "SPARSE":     "#5ba4cf",   # mid blue
    "SHELL":      "#2d7fb8",   # medium-deep blue
    "PATCHY":     "#1a5a8a",   # dark teal-blue
    "CONTINUOUS": "#0d2f4f",   # near-navy
}

# ── Condition colour map  ────────────────────────────────────────────────────
CONDITION_PALETTE = {
    "Empty":          "#cfe2f3",
    "Factin":         "#7cb9e0",
    "BranchedCortex": "#2d7fb8",
    "LinearCortex":   "#0d2f4f",
}


def set_paper_style():
    """Apply a clean, publication-ready matplotlib style."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "DejaVu Sans"],
        "pdf.fonttype":       42,
        "svg.fonttype":       "none",
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VIOLIN + SCATTER + BOX  (size distribution)
# ─────────────────────────────────────────────────────────────────────────────

def plot_violin_scatter_box(data, x_col, y_col, title, ylabel,
                             output_dir, filename):
    """
    Violin + jitter-scatter + box plot.

    Improvements vs previous version
    ─────────────────────────────────
    • Canonical condition order (Empty → Factin → Branched → Linear).
    • Box plot is drawn LAST (on top) with a white face and thick lines so it
      is always visible regardless of scatter density.
    • Each condition has its own colour from CONDITION_PALETTE.
    • Scatter alpha and size are reduced to limit overplotting.
    """
    set_paper_style()

    # ── filter & order ───────────────────────────────────────────────────────
    order   = [c for c in CONDITION_ORDER if c in data[x_col].unique()]
    labels  = [CONDITION_LABELS[c] for c in order]
    colors  = [CONDITION_PALETTE[c] for c in order]
    counts  = {c: (data[x_col] == c).sum() for c in order}

    fig, ax = plt.subplots(figsize=(10, 6))

    # ── 1. violin  ───────────────────────────────────────────────────────────
    parts = ax.violinplot(
        [data.loc[data[x_col] == c, y_col].dropna().values for c in order],
        positions=range(len(order)),
        widths=0.65,
        showmedians=False,
        showextrema=False,
    )
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor("#555555")
        body.set_alpha(0.55)
        body.set_linewidth(0.8)

    # ── 2. scatter (jitter)  ─────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    for i, (cond, col) in enumerate(zip(order, colors)):
        vals = data.loc[data[x_col] == cond, y_col].dropna().values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            i + jitter, vals,
            s=3, color=col, alpha=0.35,
            linewidths=0, zorder=2,
        )

    # ── 3. box plot ON TOP with white fill  ──────────────────────────────────
    bp = ax.boxplot(
        [data.loc[data[x_col] == c, y_col].dropna().values for c in order],
        positions=range(len(order)),
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        zorder=5,
        medianprops  =dict(color="#c0392b", linewidth=2.5),
        whiskerprops =dict(color="#333333", linewidth=1.5),
        capprops     =dict(color="#333333", linewidth=1.5),
        boxprops     =dict(facecolor="white", edgecolor="#333333", linewidth=1.5),
    )

    # ── axes labels  ─────────────────────────────────────────────────────────
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [f"{CONDITION_LABELS[c]}\nN={counts[c]:,}" for c in order],
        fontsize=11,
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(-0.6, len(order) - 0.4)

    # shade controls (first two)
    for x in range(min(2, len(order))):
        ax.axvspan(x - 0.45, x + 0.45, color="#f0f0f0", zorder=0, alpha=0.5)

    # light legend: control vs experimental
    ctrl_patch = mpatches.Patch(facecolor="#f0f0f0", edgecolor="grey",
                                 linewidth=0.6, label="Controls")
    ax.legend(handles=[ctrl_patch], loc="upper right", fontsize=9,
              frameon=False)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PHENOTYPE COMPOSITION  (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_phenotype_composition(df, x_col, phenotype_col, output_dir, filename):
    """Stacked bar chart of phenotype percentages per condition."""
    set_paper_style()

    order = [c for c in CONDITION_ORDER if c in df[x_col].unique()]
    pheno_order = ["EMPTY", "LUMENAL", "SHELL", "SPARSE", "PATCHY",
                   "CONTINUOUS", "EXCLUDED"]

    counts = (
        df.groupby([x_col, phenotype_col])
        .size()
        .unstack(fill_value=0)
        .reindex(order)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    # keep only phenotypes that actually appear
    pheno_present = [p for p in pheno_order if p in pct.columns]
    pct = pct[pheno_present]

    fig, ax = plt.subplots(figsize=(8, 5))

    bottom = np.zeros(len(order))
    for pheno in pheno_present:
        vals = pct[pheno].values
        bars = ax.bar(
            range(len(order)), vals,
            bottom=bottom,
            color=PHENOTYPE_PALETTE.get(pheno, "#aaaaaa"),
            edgecolor="white", linewidth=0.6,
            label=pheno,
        )
        # annotate segments > 5 %
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(j, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in order], fontsize=11)
    ax.set_ylabel("Percentage of Vesicles (%)", fontsize=11)
    ax.set_title("Phenotype Composition by Condition", fontsize=12, pad=8)
    ax.set_ylim(0, 108)

    handles = [mpatches.Patch(facecolor=PHENOTYPE_PALETTE.get(p, "#aaa"),
                               edgecolor="white", label=p)
               for p in pheno_present]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, frameon=False, title="Phenotype", title_fontsize=9)

    # shade controls
    for x in range(min(2, len(order))):
        ax.axvspan(x - 0.45, x + 0.45, color="#f0f0f0", zorder=0, alpha=0.45)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CORTEX MAP  (scatter: localization vs Gini)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cortex_map(cond_df, condition, output_dir):
    """5-D scatter: localization × Gini, colour = phenotype, size = t_cortex."""
    set_paper_style()

    phenotypes = cond_df["Phenotype_Category"].unique().tolist()
    # consistent ordering
    pheno_order_all = ["EMPTY","LUMENAL","SHELL","SPARSE","PATCHY","CONTINUOUS","EXCLUDED"]
    phenotypes_sorted = [p for p in pheno_order_all if p in phenotypes]

    fig, ax = plt.subplots(figsize=(7, 5))

    for pheno in phenotypes_sorted:
        sub = cond_df[cond_df["Phenotype_Category"] == pheno]
        if sub.empty:
            continue
        sizes = np.clip(sub["t_cortex"].fillna(0).values, 0, None)
        sizes = 20 + sizes * 40   # scale: min 20, grows with t_cortex

        ax.scatter(
            sub["A localization"], sub["Gini_Index"],
            c=PHENOTYPE_PALETTE.get(pheno, "#aaaaaa"),
            s=sizes, alpha=0.65, edgecolors="none",
            label=pheno, zorder=3,
        )

    ax.set_xlabel("Localization Score", fontsize=11)
    ax.set_ylabel("Gini Index (spatial heterogeneity)", fontsize=11)
    ax.set_title(f"Cortex Map: {CONDITION_LABELS.get(condition, condition)}",
                 fontsize=12)
    ax.legend(title="Phenotype", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, frameon=True, title_fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, f"Cortex_Map_{condition}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PAIR PLOT  (per condition)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pairplot(cond_df, output_dir, filename_suffix=""):
    """Seaborn pairplot of the four key cortex metrics, coloured by phenotype."""
    set_paper_style()

    metrics = ["t_cortex", "A localization", "ISM", "Gini_Index"]
    cols_present = [c for c in metrics if c in cond_df.columns]

    plot_df = cond_df[cols_present + ["Phenotype_Category"]].dropna(
        subset=cols_present
    )

    phenotypes = plot_df["Phenotype_Category"].unique().tolist()
    pheno_order_all = ["EMPTY","LUMENAL","SHELL","SPARSE","PATCHY","CONTINUOUS","EXCLUDED"]
    phenotypes_sorted = [p for p in pheno_order_all if p in phenotypes]
    palette = {p: PHENOTYPE_PALETTE.get(p, "#aaaaaa") for p in phenotypes_sorted}

    g = sns.PairGrid(
        plot_df, vars=cols_present, hue="Phenotype_Category",
        palette=palette, hue_order=phenotypes_sorted,
        diag_sharey=False,
    )
    g.map_diag(sns.kdeplot, fill=True, alpha=0.55, linewidth=1.2,
               warn_singular=False)
    g.map_offdiag(sns.scatterplot, s=12, alpha=0.5, edgecolor="none")
    g.add_legend(title="Phenotype", fontsize=9, title_fontsize=9,
                 bbox_to_anchor=(1.02, 0.5), loc="center left")

    condition_label = filename_suffix.lstrip("_")
    g.figure.suptitle(
        f"Metric Pair Plot — {CONDITION_LABELS.get(condition_label, condition_label)}",
        y=1.01, fontsize=12,
    )

    path = os.path.join(output_dir,
                        f"Correlation_Matrix_PairPlot{filename_suffix}.png")
    g.figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  -> Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SPEARMAN CORRELATION HEATMAP  (global)
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_matrix, output_dir, suffix="", title=None):
    """
    Lower-triangle Spearman heatmap.

    Improvements vs previous version
    ─────────────────────────────────
    • Figure is sized to give each cell ~50 px → annotations never overlap.
    • X-labels rotated 45° and right-aligned.
    • Annotation font size scales with number of variables.
    • Only the lower triangle is shown (upper masked) → half the clutter.
    • Diagonal is also masked (trivially == 1).
    • Columns / rows dropped if they are all-NaN in the correlation matrix.
    """
    set_paper_style()

    # ── clean: drop constant columns (corr = NaN) ────────────────────────────
    cm = corr_matrix.dropna(how="all", axis=0).dropna(how="all", axis=1)

    # ── drop nuisance identifiers that are not scientifically meaningful ─────
    drop_cols = ["Vesicle id", "xc", "yc", "Radius"]
    cm = cm.drop(columns=[c for c in drop_cols if c in cm.columns],
                 errors="ignore")
    cm = cm.drop(index=[c for c in drop_cols if c in cm.index],
                 errors="ignore")

    n = len(cm)
    cell_size = 0.70          # inches per cell
    fig_size  = max(10, n * cell_size)
    annot_fs  = max(5, min(9, 120 // n))   # shrink font as matrix grows

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

    # ── lower-triangle mask  ─────────────────────────────────────────────────
    mask = np.triu(np.ones_like(cm, dtype=bool), k=0)   # mask upper + diag

    sns.heatmap(
        cm,
        mask=mask,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_fs},
        linewidths=0.3,
        linecolor="#dddddd",
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Spearman ρ"},
    )

    ax.set_title(title if title else f"Spearman Correlation Matrix{suffix}",
                 fontsize=13, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45, ha="right", fontsize=max(7, annot_fs))
    ax.set_yticklabels(ax.get_yticklabels(),
                       rotation=0, fontsize=max(7, annot_fs))

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"Spearman_Correlation_Heatmap{suffix}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PHENOTYPE CHARACTERISTICS MATRIX  (box + strip + stats)
# ─────────────────────────────────────────────────────────────────────────────

def generate_category_panel(df, output_dir):
    """
    Grid of box + strip plots: one row per metric, one column per condition.
    Phenotype on x-axis within each panel, coloured by PHENOTYPE_PALETTE.
    """
    set_paper_style()

    metrics     = ["t_cortex", "A localization", "ISM", "Gini_Index"]
    metric_lbls = {
        "t_cortex":      "Cortex Thickness (t_cortex)",
        "A localization": "Localization Score",
        "ISM":           "ISM",
        "Gini_Index":    "Gini Index",
    }

    cond_order = [c for c in CONDITION_ORDER
                  if c in df["Category"].unique() and c != "Empty"]

    n_rows = len(metrics)
    n_cols = len(cond_order)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.5 * n_rows),
                             sharey=False)

    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for ci, cond in enumerate(cond_order):
        cond_df = df[df["Category"] == cond]

        # determine phenotype order for this condition
        pheno_all = ["LUMENAL","SHELL","SPARSE","PATCHY","CONTINUOUS"]
        pheno_present = [p for p in pheno_all
                         if p in cond_df["Phenotype_Category"].unique()]

        palette = {p: PHENOTYPE_PALETTE.get(p, "#aaaaaa") for p in pheno_present}

        for ri, metric in enumerate(metrics):
            ax = axes[ri, ci]
            sub = cond_df[["Phenotype_Category", metric]].dropna(subset=[metric])

            if sub.empty or not pheno_present:
                ax.set_visible(False)
                continue

            sns.boxplot(
                data=sub, x="Phenotype_Category", y=metric,
                hue="Phenotype_Category", hue_order=pheno_present,
                order=pheno_present, palette=palette,
                width=0.5, showfliers=False, linewidth=0.9,
                legend=False, ax=ax,
                boxprops=dict(alpha=0.7),
            )
            sns.stripplot(
                data=sub, x="Phenotype_Category", y=metric,
                hue="Phenotype_Category", hue_order=pheno_present,
                order=pheno_present, palette=palette,
                size=2.5, alpha=0.4, jitter=True,
                dodge=False, legend=False, ax=ax,
            )

            # simple significance brackets between adjacent pairs
            _add_sig_brackets(ax, sub, metric, pheno_present)

            if ri == 0:
                ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10)
            if ci == 0:
                ax.set_ylabel(metric_lbls.get(metric, metric), fontsize=9)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=8, rotation=20)
            ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Phenotype Metric Comparison", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "Phenotype_Characteristics_Matrix_Stats.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {path}")


def _add_sig_brackets(ax, df, metric, pheno_order):
    """
    Draw significance brackets above adjacent phenotype pairs.

    Key fixes vs previous version:
    ─────────────────────────────
    • Bracket base uses the 95th-percentile of EACH group's data (not the
      global max), so a handful of outliers no longer push brackets off-screen.
    • Step size is derived from the IQR of the combined data, keeping it
      proportional to the dense part of the distribution.
    • All significant brackets are collected first, then drawn with a clean
      line (ax.plot) at incrementally higher y-levels.
    • The y-axis limit is expanded after drawing so brackets are never clipped.
    """
    pairs = [(pheno_order[i], pheno_order[i + 1])
             for i in range(len(pheno_order) - 1)]

    # ── collect significant pairs ─────────────────────────────────────────
    sig = []   # list of (x1, x2, stars, per-group_max)
    for p1, p2 in pairs:
        g1 = df.loc[df["Phenotype_Category"] == p1, metric].dropna().values
        g2 = df.loc[df["Phenotype_Category"] == p2, metric].dropna().values
        if len(g1) < 3 or len(g2) < 3:
            continue
        try:
            _, pval = mannwhitneyu(g1, g2, alternative="two-sided")
        except Exception:
            continue
        if pval >= 0.05:
            continue
        stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
        # top of the two boxes = 75th-percentile of each group
        group_top = max(np.nanpercentile(g1, 95), np.nanpercentile(g2, 95))
        sig.append((pheno_order.index(p1), pheno_order.index(p2),
                    stars, group_top))

    if not sig:
        return

    # ── define step from the IQR of all visible data ──────────────────────
    all_vals = df[metric].dropna().values
    iqr = np.nanpercentile(all_vals, 75) - np.nanpercentile(all_vals, 25)
    # fallback: if IQR is zero (e.g. constant group), use 10 % of range
    if iqr < 1e-9:
        iqr = (all_vals.max() - all_vals.min()) * 0.1
    step = max(iqr * 0.55, np.nanpercentile(all_vals, 95) * 0.06)

    # ── draw brackets ─────────────────────────────────────────────────────
    # Start just above the current axes top (or the group tops, whichever is
    # higher), so brackets never overlap the box plots.
    current_top = ax.get_ylim()[1]
    base_y = max(current_top, max(s[3] for s in sig)) + step * 0.3

    for i, (x1, x2, stars, _) in enumerate(sig):
        y = base_y + step * i
        bar_y  = y
        tick_h = step * 0.15          # short descending ticks at bracket ends

        # horizontal bar + two descending ticks
        ax.plot([x1, x2],        [bar_y, bar_y],        color="black", lw=0.9)
        ax.plot([x1, x1],        [bar_y, bar_y - tick_h], color="black", lw=0.9)
        ax.plot([x2, x2],        [bar_y, bar_y - tick_h], color="black", lw=0.9)
        ax.text((x1 + x2) / 2,  bar_y + step * 0.05, stars,
                ha="center", va="bottom", fontsize=8.5, color="black")

    # expand y-axis so the topmost bracket + its text is fully visible
    new_top = base_y + step * (len(sig) + 0.6)
    ax.set_ylim(ax.get_ylim()[0], new_top)