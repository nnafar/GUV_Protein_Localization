# -*- coding: utf-8 -*-
"""
PLOT FLIM LIFETIMES
Reads FLIM tau (ns) data for Branched vs Linear cortices and plots it using
the same violin/box drawing engine and paper style as the rest of the
GUV pipeline (plotting.py). Keep this file in the same folder as
plotting.py so it can import the shared style/helpers from there.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Import the shared style + drawing engine straight from plotting.py ────
# This is the same approach plotting.py itself uses to keep one definition
# of "what a violin/box comparison looks like" instead of duplicating it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting import (
    set_paper_style,
    _get_figsize_single_col,
    _fs,
    _save_fig,
    _draw_standard_violin_box,
    _build_stat_label,
    _add_bracket,
)
import file_handling  # reused only for create_output_folder() below


# ============================================================
#  CONFIGURATION — Edit these two paths to match your setup
# ============================================================

# Where the FLIM lifetime CSV lives. Update this to the real location.
FLIM_CSV_PATH = r"M:\tnw\bn\gk\NN\2_Data-Analysis\FLIM\FLIM_Lifetimes.csv"

# Same ROOT_PATH as master_pipeline.py. create_output_folder() is the exact
# same function master_pipeline.py uses to build results_dir, so this
# resolves to the SAME "Batch_Analysis_Results" folder the rest of your
# pipeline already saves into -- not a separate copy of that path string.
ROOT_PATH = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\Output"
OUTPUT_DIR = file_handling.create_output_folder(ROOT_PATH, "Batch_Analysis_Results")
# ============================================================

# FLIM-specific category order/palette. Colors match the Branched/Linear
# hues used elsewhere in the pipeline (CONDITION_PALETTE's
# BranchedCortex/LinearCortex), so this figure sits visually consistently
# next to the other condition-comparison plots.
FLIM_ORDER = ["Branched", "Linear"]
FLIM_PALETTE = {
    "Branched": "#34487a",
    "Linear":   "#1a243d",
}


def load_flim_data(csv_path):
    """
    Loads the wide-format FLIM CSV -- one column per condition, NaN-padded
    at the bottom because Branched and Linear have different N -- and
    reshapes it to long format for plotting: columns ['Category', 'tau'].
    """
    df_wide = pd.read_csv(csv_path)
    df_long = (
        df_wide
        .melt(value_name="tau", var_name="Category")
        .dropna(subset=["tau"])
    )
    return df_long


def plot_flim_lifetimes(csv_path, output_dir, filename="Plot_FLIM_Lifetimes",
                         _as_figure=False):
    set_paper_style()

    df = load_flim_data(csv_path)
    order = [c for c in FLIM_ORDER if c in df["Category"].unique()]
    if not order:
        raise ValueError(
            f"None of {FLIM_ORDER} were found as columns in {csv_path}."
        )

    groups = [df.loc[df["Category"] == c, "tau"].dropna().values for c in order]
    colors = [FLIM_PALETTE.get(c, "#6B7B8C") for c in order]

    fig, ax = plt.subplots(figsize=_get_figsize_single_col())
    _draw_standard_violin_box(ax, groups, range(len(order)), colors, widths=0.12)
    ax.set_ylim(bottom=2.3)  # Lifetime can't be negative — always start the axis at 0

    # ── Significance bracket (Branched vs Linear) ───────────────────────
    # Same Mann-Whitney + Cliff's Delta logic used everywhere else in the
    # pipeline: stars show significance, bracket thickness encodes effect
    # size, and nothing is drawn at all if the difference is "ns".
    if len(order) == 2:
        v_a, v_b = groups[0], groups[1]
        stars, label, lw = _build_stat_label(v_a, v_b)
        if stars != "ns":
            y_top = max(v_a.max(), v_b.max())
            _add_bracket(ax, 0, 1, y_top, label, lw=lw, fontsize=_fs("annot"))

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=_fs("tick"))
    ax.set_ylabel("Fluorescence Lifetime, \u03c4 (ns)", fontsize=_fs("label"))
    sns.despine()

    plt.tight_layout()

    if _as_figure:
        return fig
    _save_fig(fig, output_dir, filename)
    print(f"Saved: {os.path.join(output_dir, filename + '.pdf')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot FLIM lifetime (tau, ns) data for Branched vs Linear cortices."
    )
    parser.add_argument(
        "csv_path", nargs="?", default=FLIM_CSV_PATH,
        help=f"Path to the FLIM lifetimes CSV file (default: {FLIM_CSV_PATH}).",
    )
    parser.add_argument(
        "-o", "--output_dir", default=OUTPUT_DIR,
        help=f"Directory to save the output PDF (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    plot_flim_lifetimes(args.csv_path, args.output_dir)