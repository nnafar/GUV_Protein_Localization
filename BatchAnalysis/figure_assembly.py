# -*- coding: utf-8 -*-
"""
FIGURE ASSEMBLY MODULE
======================
Composes individual plots into complete thesis figures (main + supplementary).

HOW IT WORKS
------------
Each plot function in plotting.py can now be called with _as_figure=True,
which returns the matplotlib Figure object instead of saving it to disk.

Here, we:
  1. Call those functions with _as_figure=True to get the Figure objects
  2. Render each Figure to a high-resolution image buffer in memory
     (think of it like taking a photo of each plot)
  3. Arrange those images in a grid using GridSpec
  4. Add panel labels (A, B, C...) and save the whole thing as a PDF

WHY NOT PURE VECTOR?
--------------------
Individual plots saved by plotting.py ARE true vector PDFs — infinitely sharp.
The assembled figures are "raster-in-PDF": the final PDF container is vector,
but the content inside is a 300 DPI image. This is standard practice for
multi-panel thesis figures and is accepted by all major journals.
If you need fully vector assembly, use Inkscape (open the individual PDFs
and arrange them manually — the quality will be identical to the source).

OUTPUT STRUCTURE
----------------
    <output_dir>/
        Figures/
            Main_Figure_1_Phenotype_Composition.pdf
            Main_Figure_2_Characteristics_Matrix.pdf
            Main_Figure_3_Actin_Comparison.pdf
            Main_Figure_4_Phenotype_Stratified.pdf
            Supp_Figure_S1_Radius.pdf
            Supp_Figure_S2_Batch_QC.pdf
            Supp_Figure_S3_Batch_Actin_Metrics.pdf
            Supp_Figure_S4_Correlation_Matrices.pdf
            Supp_Figure_S5_Lumen_Retention.pdf
"""

import io
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import plotting  # Our own plotting module

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# DPI used when rendering individual panels into memory buffers.
# 300 is standard for print-quality figures.
PANEL_RENDER_DPI = 300

# Font size for the "A", "B", "C" panel labels added to each subplot
PANEL_LABEL_FONTSIZE = 12

# ─────────────────────────────────────────────────────────────────────────────
# CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _render_to_image(fig):
    """
    Converts a matplotlib Figure into a NumPy image array without saving to disk.

    Analogy: imagine photocopying a drawing into a format you can paste
    onto a larger sheet of paper. The original drawing stays intact; you
    just capture a copy of it.

    Parameters:
        fig : a matplotlib Figure object (returned by plotting functions
              when called with _as_figure=True)

    Returns:
        img : a NumPy array representing the figure as a pixel image
    """
    buf = io.BytesIO()  # An in-memory file — like a RAM-based piece of paper
    fig.savefig(buf, format="png", dpi=PANEL_RENDER_DPI, bbox_inches="tight")
    buf.seek(0)  # Rewind to the start so we can read it back
    img = plt.imread(buf)  # Read the PNG bytes back as a pixel array
    plt.close(fig)  # Free the memory used by the original figure
    return img

def _place_image(ax, img):
    """
    Places a rendered image into an axes slot, removing all axis decorations.

    Think of this like gluing a photograph onto a blank space on a poster —
    the photograph already has all its own labels and axes; we just need
    to display it without adding extra frames or tick marks around it.
    """
    ax.imshow(img, aspect="auto", interpolation="lanczos")
    ax.axis("off")  # Hide tick marks, borders — the image is self-contained

def _add_panel_label(ax, letter):
    """
    Adds a bold letter (A, B, C...) to the top-left corner of a panel.

    This is the standard convention in scientific figures:
    each sub-panel gets a letter so the figure caption can refer to it
    clearly (e.g. "Fig. 3A shows localization score...").

    The label is placed slightly outside the axes boundaries so it doesn't
    overlap the plot content.
    """
    ax.text(
        -0.02, 1.03,          # Slightly left of and above the axes
        letter,
        transform=ax.transAxes,   # Coordinates relative to this axes (0–1)
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="right",
        fontfamily="Arial",
    )

def _save_assembled(fig, output_dir, filename_stem):
    """
    Saves an assembled multi-panel figure as a PDF.

    The PDF file format means the container is vector (text, metadata),
    even though the image content inside is high-resolution raster (300 DPI).
    This is standard for thesis submission.

    WHY THE TRY/EXCEPT?
    On Windows, PDF viewers (Adobe Acrobat, Edge, etc.) lock the file while
    it is open. If you re-run the pipeline without closing the viewer first,
    Python gets a PermissionError. The except block catches this and prints
    a clear instruction instead of crashing the whole pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename_stem + ".pdf")
    try:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {filename_stem}.pdf")
    except PermissionError:
        plt.close(fig)
        print(f"    ⚠ Could not save {filename_stem}.pdf")
        print(f"      → Close the file in your PDF viewer and re-run.")

# ─────────────────────────────────────────────────────────────────────────────
# BATCH LABEL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_batch_label(df):
    """
    Guarantees the Batch_Label column exists before S2/S3 try to use it.

    ROOT CAUSE OF THE SKIP:
    analysis_batch.py creates Batch_Label on a local copy of df called
    df_labelled, but never writes it back to the original df.  So the df
    that reaches figure_assembly is missing the column entirely.

    FIX:
    We recreate it here from Batch_ID + Category using the same shortening
    logic as _shorten_batch_label() in analysis_batch.py:
        '260208_BranchedCortex_1'  →  '260208 #1'

    Returns a copy of df with Batch_Label added (or the original df
    unchanged if the column already exists or source columns are missing).
    """
    if "Batch_Label" in df.columns:
        return df  # Already there — nothing to do

    if "Batch_ID" not in df.columns or "Category" not in df.columns:
        # Can't recreate the column — S2/S3 will still skip gracefully
        return df

    def _shorten(batch_id, condition):
        # Strip the condition name from the folder name, then parse date + run number
        # e.g. '260208_BranchedCortex_1' + 'BranchedCortex' → '260208 #1'
        short = str(batch_id).replace(str(condition), "").replace("__", "_").strip("_")
        parts  = [p for p in short.split("_") if p]
        if len(parts) >= 2:
            return f"{parts[0]} #{parts[-1]}"   # '260208 #1'
        elif len(parts) == 1:
            return parts[0]
        else:
            return str(batch_id)                # Fallback — use the full name

    df = df.copy()  # Don't modify the caller's DataFrame
    df["Batch_Label"] = df.apply(
        lambda row: _shorten(row["Batch_ID"], row["Category"]), axis=1
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def assemble_main_figure_1(df, output_dir):
    """
    MAIN FIGURE 1 — Phenotype Composition

    Single-panel figure showing the percentage breakdown of phenotypes
    (EMPTY, LUMENAL, SPARSE, PATCHY, CONTINUOUS, EXCLUDED) across all
    four experimental conditions.

    This is the opening figure of the thesis results: it answers
    "what did we actually make?" before anything else.
    """
    print("  Assembling Main Figure 1: Phenotype Composition...")

    if "Condition" in df.columns:
        x_col = "Condition"
    elif "Category" in df.columns:
        x_col = "Category"
    else:
        print("    Skipped: no condition column found.")
        return

    if "Phenotype_Category" not in df.columns:
        print("    Skipped: Phenotype_Category column not found.")
        return

    # Call the plot function and get back the figure without saving it yet
    fig = plotting.plot_phenotype_composition(
        df, x_col, "Phenotype_Category",
        output_dir=None,  # Don't save yet — we want the figure object
        filename="placeholder",
        _as_figure=True,
    )
    if fig is None:
        return

    img = _render_to_image(fig)

    # Create the assembled figure with a single panel
    assembled = plt.figure(figsize=(8, 5.5))
    ax = assembled.add_subplot(1, 1, 1)
    _place_image(ax, img)
    _add_panel_label(ax, "A")

    _save_assembled(assembled, output_dir, "Main_Figure_1_Phenotype_Composition")


def assemble_main_figure_2(df, output_dir):
    """
    MAIN FIGURE 2 — Phenotype Characteristics Matrix

    Multi-panel figure showing how each phenotype differs on three
    biological metrics: cortex thickness, localization score, Gini index.

    This is the validation figure: it proves the phenotype labels
    correspond to real, measurable biological differences.
    """
    print("  Assembling Main Figure 2: Characteristics Matrix...")

    if "Phenotype_Category" not in df.columns:
        print("    Skipped: Phenotype_Category column not found.")
        return

    fig = plotting.generate_category_panel(df, output_dir=None, _as_figure=True)
    if fig is None:
        return

    img = _render_to_image(fig)

    assembled = plt.figure(figsize=(10, 9))
    ax = assembled.add_subplot(1, 1, 1)
    _place_image(ax, img)
    _add_panel_label(ax, "A")

    _save_assembled(assembled, output_dir, "Main_Figure_2_Characteristics_Matrix")


def assemble_main_figure_3(df, output_dir):
    """
    MAIN FIGURE 3 — Three-Way Actin Comparison

    Two-panel figure combining:
      A  The three-way actin comparison across F-actin, Branched, Linear
      B  Lumen retention across the same three conditions

    Together these panels tell the core biological story: how branched
    and linear cortices differ from free actin and from each other.
    """
    print("  Assembling Main Figure 3: Actin Comparison...")

    has_actin_cols = any(c in df.columns for c in ["A localization", "Gini_Index", "t_cortex"])
    if not has_actin_cols:
        print("    Skipped: actin metric columns not found.")
        return

    # Panel A: three-way comparison
    fig_a = plotting.plot_actin_three_way(df, output_dir=None, _as_figure=True)
    # Panel B: lumen retention (optional — skip gracefully if data is absent)
    fig_b = None
    if "A Lumen/Bg" in df.columns:
        fig_b = plotting.plot_lumen_retention(df, output_dir=None, _as_figure=True)

    img_a = _render_to_image(fig_a) if fig_a is not None else None
    img_b = _render_to_image(fig_b) if fig_b is not None else None

    if img_b is not None:
        # Panel A is a 2-row grid (approx square); Panel B is a single tall plot.
        # Width ratio 3:1 keeps B from being stretched horizontally.
        # Height is driven by Panel A's native aspect (ncols=2, nrows=2 → ~square).
        assembled = plt.figure(figsize=(13, 9))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.06)
        ax_a = assembled.add_subplot(gs[0])
        ax_b = assembled.add_subplot(gs[1])
        _place_image(ax_a, img_a)
        _place_image(ax_b, img_b)
        _add_panel_label(ax_a, "A")
        _add_panel_label(ax_b, "B")
    else:
        # Lumen data unavailable — single-panel figure
        assembled = plt.figure(figsize=(10, 5.5))
        ax_a = assembled.add_subplot(1, 1, 1)
        _place_image(ax_a, img_a)
        _add_panel_label(ax_a, "A")

    _save_assembled(assembled, output_dir, "Main_Figure_3_Actin_Comparison")


def assemble_main_figure_4(df, output_dir):
    """
    MAIN FIGURE 4 — Phenotype-Stratified Comparison (CONTINUOUS only)

    Single assembled figure showing the comparison between Branched and
    Linear cortices restricted to CONTINUOUS phenotype vesicles only.

    This controls for phenotype distribution differences between conditions,
    making the comparison methodologically rigorous — a key strength to
    highlight in the thesis.
    """
    print("  Assembling Main Figure 4: Phenotype-Stratified Comparison...")

    if "Phenotype_Category" not in df.columns:
        print("    Skipped: Phenotype_Category column not found.")
        return

    fig = plotting.plot_phenotype_stratified(df, output_dir=None, _as_figure=True)
    if fig is None:
        return

    img = _render_to_image(fig)

    assembled = plt.figure(figsize=(9, 8))
    ax = assembled.add_subplot(1, 1, 1)
    _place_image(ax, img)
    _add_panel_label(ax, "A")

    _save_assembled(assembled, output_dir, "Main_Figure_4_Phenotype_Stratified")


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def assemble_supp_figure_S1(df, output_dir):
    """
    SUPPLEMENTARY FIGURE S1 — GUV Radius Across Conditions

    Shows that vesicle size is comparable across conditions, ruling out
    the possibility that any biological differences are driven by size.
    This is a quality-control figure — important for credibility but not
    the main scientific claim.
    """
    print("  Assembling Supplementary Figure S1: Radius...")

    if "Refined Radius (um)" not in df.columns:
        print("    Skipped: Refined Radius column not found.")
        return

    fig = plotting.plot_cortex_vs_empty_radius(df, output_dir=None, _as_figure=True)
    if fig is None:
        return

    img = _render_to_image(fig)

    assembled = plt.figure(figsize=(7, 5))
    ax = assembled.add_subplot(1, 1, 1)
    _place_image(ax, img)
    _add_panel_label(ax, "A")

    _save_assembled(assembled, output_dir, "Supp_Figure_S1_Radius")


def assemble_supp_figure_S2(df, output_dir):
    """
    SUPPLEMENTARY FIGURE S2 — Batch Quality Control

    Two-panel figure:
      A  Vesicle size distribution per batch (reproducibility check)
      B  Phenotype composition per batch (stability check)

    Together these show that the results were consistent across independent
    experimental preparations — the backbone of reproducibility claims.
    """
    print("  Assembling Supplementary Figure S2: Batch QC...")

    df = _ensure_batch_label(df)  # Recreate Batch_Label if analysis_batch did not persist it
    if "Batch_Label" not in df.columns:
        print("    Skipped: Batch_Label and Batch_ID columns not found.")
        return

    fig_a = plotting.plot_batch_size_distribution(df, output_dir=None, _as_figure=True)
    fig_b = None
    if "Phenotype_Category" in df.columns:
        fig_b = plotting.plot_batch_phenotype_composition(df, output_dir=None, _as_figure=True)

    img_a = _render_to_image(fig_a) if fig_a is not None else None
    img_b = _render_to_image(fig_b) if fig_b is not None else None

    if img_a is None:
        print("    Skipped: no data available.")
        return

    if img_b is not None:
        # Stack panels vertically: size on top, phenotype composition below
        # Larger figsize so batch labels and N values remain readable
        assembled = plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(2, 1, hspace=0.08)
        ax_a = assembled.add_subplot(gs[0])
        ax_b = assembled.add_subplot(gs[1])
        _place_image(ax_a, img_a)
        _place_image(ax_b, img_b)
        _add_panel_label(ax_a, "A")
        _add_panel_label(ax_b, "B")
    else:
        assembled = plt.figure(figsize=(12, 5))
        ax_a = assembled.add_subplot(1, 1, 1)
        _place_image(ax_a, img_a)
        _add_panel_label(ax_a, "A")

    _save_assembled(assembled, output_dir, "Supp_Figure_S2_Batch_QC")


def assemble_supp_figure_S3(df, output_dir):
    """
    SUPPLEMENTARY FIGURE S3 — Batch Actin Metrics

    Shows that the key actin measurements (localization, Gini index,
    cortex thickness) were stable across batches within each condition.

    This rules out the possibility that the main figure results are
    driven by one lucky experiment.
    """
    print("  Assembling Supplementary Figure S3: Batch Actin Metrics...")

    df = _ensure_batch_label(df)  # Recreate Batch_Label if analysis_batch did not persist it
    has_actin = any(c in df.columns for c in ["A localization", "Gini_Index", "t_cortex"])
    if not has_actin or "Batch_Label" not in df.columns:
        print("    Skipped: required columns not found.")
        return

    fig = plotting.plot_batch_actin_metrics(df, output_dir=None, _as_figure=True)
    if fig is None:
        return

    img = _render_to_image(fig)

    # Larger canvas so metric labels, batch labels and N values are readable
    assembled = plt.figure(figsize=(20, 13))
    ax = assembled.add_subplot(1, 1, 1)
    _place_image(ax, img)
    _add_panel_label(ax, "A")

    _save_assembled(assembled, output_dir, "Supp_Figure_S3_Batch_Actin_Metrics")


def assemble_supp_figure_S4(df, output_dir):
    """
    SUPPLEMENTARY FIGURE S4 — Correlation Matrices (one per condition)

    A 2×2 grid of pair plots, one for each actin-bearing condition.
    Each pair plot shows how the biological metrics relate to each other
    within that condition, coloured by phenotype.

    This is the exploratory deep-dive: it doesn't make a specific claim
    but lets readers (and your committee) examine the data structure.
    """
    print("  Assembling Supplementary Figure S4: Correlation Matrices...")

    # Resolve which column holds the condition name — check column names directly.
    # Important: never use 'or' between two pandas Series — that raises a
    # ValueError ("The truth value of a Series is ambiguous") because pandas
    # doesn't know whether you mean any(), all(), or something else.
    # Checking df.columns (a plain list) is always safe.
    if "Category" in df.columns:
        cat_col = "Category"
    elif "Condition" in df.columns:
        cat_col = "Condition"
    else:
        print("    Skipped: no condition column found.")
        return

    conditions_for_pairplot = [
        c for c in ["Factin", "BranchedCortex", "LinearCortex"]
        if c in df[cat_col].unique()
    ]

    if not conditions_for_pairplot:
        print("    Skipped: no matching conditions found.")
        return

    # Render one pair plot per condition
    imgs = []
    for cond in conditions_for_pairplot:
        cond_df = df[df[cat_col] == cond].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plotting.plot_pairplot(
                cond_df,
                output_dir=None,
                filename_suffix=f"_{cond}",
                _as_figure=True,
            )
        if fig is not None:
            imgs.append((cond, _render_to_image(fig)))

    if not imgs:
        print("    Skipped: no pair plots could be generated.")
        return

    n = len(imgs)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols  # Ceiling division

    # Each pair plot is intrinsically large (4×4 subplots); give each panel
    # enough canvas space so axis labels and correlation values are legible
    assembled = plt.figure(figsize=(ncols * 10, nrows * 10))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.08, wspace=0.08)
    labels = "ABCDEFGH"  # Enough letters for up to 8 panels

    for i, (cond, img) in enumerate(imgs):
        row, col = divmod(i, ncols)
        ax = assembled.add_subplot(gs[row, col])
        _place_image(ax, img)
        _add_panel_label(ax, labels[i])
        # Condition name is already the title inside the pairplot figure itself;
        # no need to add it again here.

    # Hide any unused grid cells (if n is odd with ncols=2)
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        assembled.add_subplot(gs[row, col]).set_visible(False)

    _save_assembled(assembled, output_dir, "Supp_Figure_S4_Correlation_Matrices")


def assemble_supp_figure_S5(df, output_dir):
    """
    SUPPLEMENTARY FIGURE S5 — Lumen Actin Retention

    Shows how much actin signal is present in the vesicle lumen
    (compared to the background) across the three actin conditions.

    This supports the main figures by showing that lumenal actin
    is accounted for and does not confound the surface localization results.
    """
    print("  Assembling Supplementary Figure S5: Lumen Retention...")

    if "A Lumen/Bg" not in df.columns:
        print("    Skipped: A Lumen/Bg column not found.")
        return

    fig = plotting.plot_lumen_retention(df, output_dir=None, _as_figure=True)
    if fig is None:
        return

    img = _render_to_image(fig)

    assembled = plt.figure(figsize=(6, 5))
    ax = assembled.add_subplot(1, 1, 1)
    _place_image(ax, img)
    _add_panel_label(ax, "A")

    _save_assembled(assembled, output_dir, "Supp_Figure_S5_Lumen_Retention")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_figure_assembly(df, base_output_dir):
    """
    Runs the full figure assembly pipeline — main figures + supplementary.

    Call this from master_pipeline.py after all analysis steps are complete,
    passing in the combined dataframe (all conditions merged) and the base
    output directory.

    Parameters:
        df             : the merged DataFrame containing all conditions,
                         with columns including Category, Phenotype_Category,
                         Refined Radius (um), Batch_Label, and actin metrics.
        base_output_dir: the root output folder (e.g. "Output/").
                         A "Figures/" subfolder will be created inside it.
    """
    figures_dir = os.path.join(base_output_dir, "Figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("FIGURE ASSEMBLY")
    print("=" * 60)
    print(f"Output folder: {figures_dir}\n")

    # ── Main Figures ──────────────────────────────────────────────
    print("[ Main Figures ]")
    assemble_main_figure_1(df, figures_dir)
    assemble_main_figure_2(df, figures_dir)
    assemble_main_figure_3(df, figures_dir)
    assemble_main_figure_4(df, figures_dir)

    # ── Supplementary Figures ─────────────────────────────────────
    print("\n[ Supplementary Figures ]")
    assemble_supp_figure_S1(df, figures_dir)
    assemble_supp_figure_S2(df, figures_dir)
    assemble_supp_figure_S3(df, figures_dir)
    assemble_supp_figure_S4(df, figures_dir)
    assemble_supp_figure_S5(df, figures_dir)

    print("\n" + "=" * 60)
    print("Figure assembly complete.")
    print("=" * 60)