# -*- coding: utf-8 -*-
"""
main.py

Main script for GUV protein-localization analysis.

CHANGES FROM PREVIOUS VERSION:
-------------------------------
1. ANALYSIS_CONFIG dictionary added.
   To tune your analysis, you only need to edit this file.

2. Per-vesicle threshold.
   Previously a single threshold was computed from the FIRST vesicle
   and reused for every vesicle in the image. This is unreliable when
   vesicles vary in brightness.
   Now each worker thread computes its own threshold from its own
   vesicle's local image region. The global pre-computation is removed.

   IMPORTANT: threshold_method_manual=True requires interactive input
   from the user (it shows a plot and asks you to type a choice).
   Interactive input does NOT work inside a parallel loop.
   If you set threshold_method_manual=True, you MUST also set n_jobs=1
   to run sequentially.  The code will warn you if this is not the case.

3. ANALYSIS_CONFIG is passed as an extra argument to
   process_single_vesicle, which passes it further to the shape
   analysis functions.
"""

# --- Stop plots from displaying in Spyder ---
import matplotlib
matplotlib.use('Agg')
# --------------------------------------------

import numpy as np
import skeleton as skl
from joblib import Parallel, delayed
import csv
import os

# ===========================================================================
#  INPUT PATHS
# ===========================================================================

## Specify paths to the directories containing the data:
path_membrane   = r"D:\ProteinLocalization\4.Factin\260227_Factin_2\ImageSequence\C1"
path_detected   = r"D:\ProteinLocalization\4.Factin\260227_Factin_2\ImageSequence\C1\detected"
path_septin     = r""
path_actin      = r"D:\ProteinLocalization\4.Factin\260227_Factin_2\ImageSequence\C3"

# Where to save all output files and plots:
path_to_output_root = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\Output"

# A unique name for this specific dataset
# Change this every time you point at a different data folder.
dataset_name = "260227_Factin_2"  

# ===========================================================================
#  WHICH PROTEINS ARE PRESENT?
# ===========================================================================

Septin = False
Actin  = True

# ===========================================================================
#  ANALYSIS CONFIGURATION
# ===========================================================================
# All tunable parameters in one place.
# Change values here; the rest of the code reads them automatically.
# -----------------------------------------------------------------------------

ANALYSIS_CONFIG = {

    # ------------------------------------------------------------------
    # PROFILE EXTRACTION
    # ------------------------------------------------------------------

    # Number of angular directions to sample around each vesicle.
    # 360 = one profile per degree (good balance of speed and detail).
    'num_angles': 360,

    # How far out (relative to each vesicle's radius) to extend the
    # radial profiles.  1.2 = profiles go 20% beyond the membrane.
    'length_excess': 1.2,

    # Step size between sample points along the radial direction (pixels).
    # dr=1 means sample every pixel. dr=2 would halve the array length.
    'dr': 1,

    # How large a square region (in units of the vesicle radius) to use
    # when estimating background signal outside the membrane.
    'size_mask': 3,

    # How large a square region to use when plotting the zoomed vesicle view.
    # Kept equal to length_excess so the plot matches the profile extent.
    'size_view': 1.2,   # keep equal to length_excess

    # Radius of the central lumen area (in units of the vesicle radius)
    # used when computing how much protein is inside vs on the membrane.
    'size_central_area': 0.25,   # = 1/4 of radius

    # Remove this many pixels at the very centre of each radial profile
    # before analysis.  The centre is often artificially bright due to
    # interpolation artefacts in map_coordinates.
    'pixels_to_remove_centre': 2,

    # ------------------------------------------------------------------
    # VESICLE FILTERING
    # ------------------------------------------------------------------

    # Vesicles with a refined radius smaller than this (in µm) are treated
    # as debris or detection errors and excluded from analysis.
    'min_vesicle_radius_um': 2.5,

    # ------------------------------------------------------------------
    # BACKGROUND THRESHOLD
    # ------------------------------------------------------------------

    # False = automatic Li threshold, computed separately for each vesicle
    #         from that vesicle's own local image region.  Safe for parallel.
    # True  = interactive: shows a plot of threshold methods and asks you to
    #         type a choice.  *** REQUIRES n_jobs = 1 (see below) ***
    'threshold_method_manual': False,

    # ------------------------------------------------------------------
    # SHAPE ANALYSIS — sector-based
    # ------------------------------------------------------------------

    # Number of pie-slice sectors to cut the vesicle into.
    # More sectors = finer spatial resolution, but noisier.
    # 8 is a good default.
    'num_sectors': 8,

    # A peak in the radial profile must be at least this FRACTION of the
    # sector maximum to be counted as a real membrane peak.
    # 0.3 = must be at least 30% of the brightest point in that sector.
    'sector_peak_height_fraction': 0.3,

    # A peak must also exceed this ABSOLUTE signal level.
    # This prevents noise spikes in very dim sectors from being counted
    # as membrane peaks (which would falsely inflate clustering risk).
    # Set this to ~2-3x your camera's typical background noise level.
    'sector_peak_min_signal': 10.0,

    # Two peaks must be at least this many pixels apart to be counted
    # as two separate peaks (rather than two sides of the same peak).
    'sector_peak_min_distance_px': 3,


    # ------------------------------------------------------------------
    # SHAPE ANALYSIS — Solidity thresholds
    # ------------------------------------------------------------------
    # Solidity = polygon_area / convex_hull_area  (computed from sector radii)
    #   1.0   = perfectly convex (ideal sphere)
    #   < 1.0 = concavities: dents, budding, touching vesicles
    #
    # 'good'         → solidity ≥ solidity_good_threshold
    # 'acceptable'   → solidity ≥ solidity_acceptable_threshold
    # 'questionable' → solidity <  solidity_acceptable_threshold
    #                  (triggers a comment in the CSV)
    'solidity_good_threshold':        0.95,
    'solidity_acceptable_threshold':  0.85,

    # ------------------------------------------------------------------
    # OUTPUT OPTIONS
    # ------------------------------------------------------------------

    # Show an overview image with all detected vesicle centres marked.
    'plot_detected_centres': True,

    # Show the background mask for each vesicle (slow, useful for debugging).
    'plot_mask': False,

    # Save radial and angular intensity profile plots for each vesicle.
    'plot_int_profiles': True,

    # Save standalone, clean membrane/actin crop images (scale bar, no
    # text) for each vesicle. Independent of 'plot_int_profiles' above —
    # these are what the batch-level "representative channel images"
    # comparison figure pulls from later, so leave this True even if you
    # turn 'plot_int_profiles' off to save time/disk space.
    'save_channel_crops': True,
}

# ===========================================================================
#  PARALLEL PROCESSING
# ===========================================================================

# -1 = use all available CPU cores (fastest).
#  1 = run one vesicle at a time (required for threshold_method_manual=True).
n_jobs = -1

# Safety check: warn the user if they've asked for interactive thresholding
# in a parallel run — that combination will deadlock or crash.
if ANALYSIS_CONFIG['threshold_method_manual'] and n_jobs != 1:
    print("=" * 60)
    print("WARNING: threshold_method_manual=True requires n_jobs=1.")
    print("Interactive input does not work in parallel workers.")
    print("Automatically setting n_jobs=1 for this run.")
    print("=" * 60)
    n_jobs = 1

# ===========================================================================
#  PREPARATION — runs once before the per-vesicle loop
# ===========================================================================

proteins_present = np.array((Septin, Actin), dtype=bool)

# Package profile parameters into an array (skeleton.py uses index access)
parameters_profiles = np.array((
    ANALYSIS_CONFIG['num_angles'],
    ANALYSIS_CONFIG['length_excess'],
    ANALYSIS_CONFIG['dr'],
))

# Package size parameters into an array
parameters_sizes = np.array((
    ANALYSIS_CONFIG['size_mask'],
    ANALYSIS_CONFIG['size_view'],
    ANALYSIS_CONFIG['size_central_area'],
))

# Scan the membrane folder to find all experiment sets to process
exp_info_all_sets = skl.find_projects_info(path_membrane)

# Ensure we always have a 2D array, even if only one set was found
if exp_info_all_sets.ndim == 1:
    num_sets = 1
    exp_info_all_sets = exp_info_all_sets.reshape(1, -1)
else:
    num_sets = len(exp_info_all_sets[:, 0])

# ===========================================================================
#  MAIN LOOP — one iteration per experiment set (image region)
# ===========================================================================

for s in range(num_sets):

    exp_info = exp_info_all_sets[s, :]
    print(f"--- Processing Set {s+1}/{num_sets}: {exp_info[2]} ---")

    # Create folder structure: Output / Date_Experiment / Region_ID
    final_output_path = skl.create_directory_structure(path_to_output_root, exp_info, dataset_name)

    # Create the CSV file and write its header row
    output_csv_path = skl.create_output_file(final_output_path, proteins_present)

    # NEW: Create the per-vesicle radial intensity profile CSV.
    # This stores the raw radially-averaged membrane/actin curves (one row
    # per radius point per vesicle) that the Int_prof_Radial plots are
    # drawn from — useful for re-plotting or pooling profiles later
    # without having to re-run the whole pipeline.
    radial_csv_path = skl.create_radial_profile_csv(final_output_path)

    # Read image channels and vesicle coordinates
    channels_data, coordinates, pixel_size = skl.read_files(
        path_membrane, path_septin, path_actin, path_detected,
        exp_info, proteins_present
    )

    if channels_data is None:
        print(f"    → Skipped (no detection data)\n")
        continue

    # Register custom colormaps (cyan, magenta, yellow)
    skl.color_maps()

    num_vesicles = len(coordinates[:, 0])

    # image_dim stores (X_size, Y_size) — used for boundary checking
    image_dim = np.array((channels_data.shape[1], channels_data.shape[0]))

    # Optional overview plot showing all detected vesicle centres
    skl.detected_centres(
        ANALYSIS_CONFIG['plot_detected_centres'],
        num_vesicles,
        channels_data[:, :, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        final_output_path,
        exp_info
    )

    # NOTE: The single global threshold calculation that was here before
    # has been REMOVED. Each vesicle now computes its own threshold inside
    # process_single_vesicle using its own local image region.
    # This gives more accurate background masks for each individual vesicle.

    all_refined_radii = []

    # -----------------------------------------------------------------------
    #  PARALLEL PROCESSING — each vesicle is processed independently
    # -----------------------------------------------------------------------

    print(f"Starting parallel processing for {num_vesicles} vesicles "
          f"(n_jobs={n_jobs})...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(skl.process_single_vesicle)(
            coordinates[i, :],
            channels_data,
            image_dim,
            parameters_profiles,
            proteins_present,
            ANALYSIS_CONFIG['size_mask'],
            final_output_path,
            exp_info,
            parameters_sizes,
            ANALYSIS_CONFIG['plot_int_profiles'],
            ANALYSIS_CONFIG['plot_mask'],
            pixel_size,
            ANALYSIS_CONFIG,
            ANALYSIS_CONFIG['save_channel_crops'],
        )
        for i in range(num_vesicles)
    )

    # -----------------------------------------------------------------------
    #  COLLECT RESULTS AND WRITE TO CSV
    # -----------------------------------------------------------------------

    print("Saving results...")

    # Open in append mode — the header was already written by create_output_file
    with open(output_csv_path, "a", newline='') as output_file:
        writer = csv.writer(output_file)

        # NEW: also open the radial profile CSV in append mode, so we can
        # write every vesicle's radial-profile rows right alongside its
        # Analysis_Results.csv summary row, in the same pass over `results`.
        with open(radial_csv_path, "a", newline='') as radial_file:
            radial_writer = csv.writer(radial_file)

            for row, refined_radius, radial_profile_rows in results:
                writer.writerow(row)
                if refined_radius is not None and refined_radius > 0:
                    all_refined_radii.append(refined_radius)

                # radial_profile_rows is a LIST of rows (one per radius
                # point), so we use writerows (plural) instead of writerow.
                # It's None for vesicles that hit the very first "margins"
                # exit, before any radial profile existed yet.
                if radial_profile_rows is not None:
                    radial_writer.writerows(radial_profile_rows)

    # Plot histogram of vesicle sizes
    skl.plot_size_distribution(all_refined_radii, final_output_path)
    print("Done with set!\n")

print("All sets processed.")