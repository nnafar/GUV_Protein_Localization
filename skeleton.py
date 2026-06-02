# -*- coding: utf-8 -*-
"""
skeleton.py

Contains all necessary functions for the GUV analysis pipeline.

CHANGES IN THIS VERSION:
-------------------------
- Automatic pixel size detection (tifffile)
- Robust localization calculation (np.max)
- ADVANCED ACTIN METRICS: ISM, Gini Index, t_cortex
- Physical unit plotting (microns)
- GEOMETRIC GATING: Excludes 'wide_peak' artifacts automatically
- FIX: Defensive directory creation to prevent FileNotFoundError on network drives
- FIX: Proper NaN handling for excluded vesicles to maintain CSV alignment
- NEW: Sector-Based Shape Analysis + Deformability Score
- FIX: plt.close() added to detected_centres (was causing a memory leak)
- FIX: Duplicate CV calculation removed from check_membrane_quality
- FIX: define_threshold now takes analysis_config instead of a bare user_choice bool
- FIX: process_single_vesicle computes per-vesicle threshold (Observation 3)
- FIX: along_radius_full saved before slicing and passed to shape analysis (Bug 3)
- FIX: lumen_val initialised at top of process_single_vesicle (prevents NameError)
- FIX: analysis_config passed through to shape analysis (Design Issues 1/2/3)

@author: kkourkoulou (Updated by Gemini, reviewed and fixed by Claude)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import csv

from skimage.filters import try_all_threshold
from skimage.filters import threshold_li, threshold_otsu, threshold_yen, \
                            threshold_isodata, threshold_mean, threshold_minimum, \
                            threshold_triangle
from skimage.transform import hough_circle, hough_circle_peaks

from scipy import ndimage as nd
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import map_coordinates
from scipy.stats import kurtosis
import cv2
import tifffile
import warnings

from guve_shape_analysis import (
    analyze_vesicle_sectors,
    calculate_deformability_score,
    plot_sector_analysis,
    format_shape_metrics_for_csv
)


# =============================================================================
# EXPERIMENT / FILE UTILITIES
# =============================================================================

def find_projects_info(path_membrane):
    """
    Finds the experiment information of all data sets to be analysed
    by scanning the Membrane (C1) directory.
    """
    set_of_data = 0
    exp_info_all_sets = np.zeros((4), dtype=str)

    for f in os.listdir(path_membrane):
        if f.endswith("C1.tif"):
            name, extension = os.path.splitext(f)
            parts = name.split("-")
            parts.pop(-1)
            full_name = "-".join(parts)
            exp_info_single_set = [parts[0], parts[1], full_name, full_name]

            if set_of_data == 0:
                exp_info_all_sets = np.array(exp_info_single_set)
            else:
                exp_info_all_sets = np.vstack((exp_info_all_sets, exp_info_single_set))

            set_of_data += 1

    return exp_info_all_sets


def create_directory_structure(path_to_output_root, exp_info, dataset_name=""):
    """
    Creates a hierarchical folder structure:
      Root -> Dataset_Name -> Region_ID
    
    Example result:
      Output/260208_BranchedCortex_1/Region0000/
      Output/260208_BranchedCortex_2/Region0000/
    """
    region_id = exp_info[1]

    final_output_path = os.path.join(
        path_to_output_root,
        dataset_name,
        region_id
    )

    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    return final_output_path


def read_files(path_membrane, path_septin, path_actin, path_detected,
               exp_info, proteins_present):
    """
    Reads image files and vesicle detection CSV from their respective folders.
    Detects pixel size from TIFF metadata.
    """
    file_stem = exp_info[3]

    # 1. Read Membrane (C1) — always present
    c1_path = os.path.join(path_membrane, file_stem + "-C1.tif")
    channels_data = plt.imread(c1_path)

    # --- Detect pixel size from metadata ---
    pixel_size = 0.09  # fallback default (µm per pixel)

    try:
        with tifffile.TiffFile(c1_path) as tif:
            imagej_metadata = tif.imagej_metadata
            if imagej_metadata and 'spacing' in imagej_metadata:
                pixel_size = float(imagej_metadata['spacing'])
            elif tif.pages[0].tags.get('XResolution'):
                x_res = tif.pages[0].tags['XResolution'].value
                if tif.pages[0].tags.get('ResolutionUnit').value == 3:
                    pixel_size = 10000.0 / (x_res[0] / x_res[1])
    except Exception:
        print(f"Warning: Could not detect pixel size for {file_stem}. "
              f"Using default {pixel_size}.")

    print(f"Processing: {file_stem} (Pixel Size: {pixel_size:.4f} µm)")

    # 2. Read Septin (C2) — if present
    if proteins_present[0]:
        c2_path = os.path.join(path_septin, file_stem + "-C2.tif")
        if os.path.exists(c2_path):
            channels_data = np.dstack((channels_data, plt.imread(c2_path)))
        else:
            print(f"Warning: Septin file not found at {c2_path}")

    # 3. Read Actin (C3) — if present
    if proteins_present[1]:
        c3_path = os.path.join(path_actin, file_stem + "-C3.tif")
        if os.path.exists(c3_path):
            channels_data = np.dstack((channels_data, plt.imread(c3_path)))
        else:
            print(f"Warning: Actin file not found at {c3_path}")

    # Safety: ensure 3D even if only one channel loaded
    if channels_data.ndim == 2:
        channels_data = channels_data[:, :, np.newaxis]

    # 4. Read detected vesicles CSV
    csv_path = os.path.join(path_detected, file_stem + "-detected_vesicles.csv")

    if not os.path.exists(csv_path):
        print(f"  SKIPPING {file_stem}: No detected_vesicles.csv found")
        return None, None, None

    try:
        detected_vesicles = pd.read_csv(csv_path)
    except Exception:
        print(f"  SKIPPING {file_stem}: Error reading CSV")
        return None, None, None

    if len(detected_vesicles) == 0:
        print(f"  SKIPPING {file_stem}: CSV is empty")
        return None, None, None

    coordinates = np.zeros((len(detected_vesicles), 4))
    coordinates[:, 0] = np.arange(1, len(detected_vesicles) + 1)
    coordinates[:, 1] = detected_vesicles.iloc[:, 0].values       # x
    coordinates[:, 2] = detected_vesicles.iloc[:, 1].values       # y
    coordinates[:, 3] = detected_vesicles.iloc[:, 2].values / 2   # diameter → radius

    return channels_data, coordinates, pixel_size


# =============================================================================
# CSV OUTPUT
# =============================================================================

def create_output_file(final_output_path, proteins_present):
    """
    Creates an empty output CSV file with the correct column headers.

    FIX: The fallback header list (used when no proteins are present, e.g.
    the Empty condition) previously only contained ["M Background"].
    This meant that columns computed purely from the MEMBRANE channel —
    specifically 'Refined Radius (um)', 'Comment', and all shape analysis
    metrics — were never written to the CSV for Empty vesicles.

    This is incorrect because:
      - The refined radius comes from refine_guv_center(), which runs
        for every vesicle regardless of protein content.
      - Shape analysis (sector-based deformability) also only uses the
        membrane channel (intensity_profiles[:, :, 0]).
      - Both are important quality metrics even when no protein is present.

    The fix adds these membrane-derived columns to all header variants,
    including the fallback case.
    """
    protein_status = plot_format(proteins_present)

    base_headers = ["Date", "Name", "Image", "Vesicle id", "xc", "yc", "Radius"]

    # These columns are ALWAYS computed, regardless of which proteins are
    # present.  They come from the membrane channel only.
    membrane_only_headers = [
        "Solidity", "Shape_Quality_Flag", "Sector_Details",
        "Refined Radius (um)", "Comment",
    ]

    if protein_status == "both_proteins":
        specific_headers = [
            "M Background", "S Background", "A Background",
            "S localization", "A localization",
            "S Lumen", "A Lumen", "S Lumen/Bg", "A Lumen/Bg",
            "t_cortex", "ISM", "Gini_Index", "Radial_Kurtosis",
        ]

    elif protein_status == "only_septin":
        specific_headers = [
            "M Background", "S Background", "S localization",
            "S Lumen", "S Lumen/Bg",             
            "t_cortex", "ISM", "Gini_Index", "Radial_Kurtosis",
        ]

    elif protein_status == "only_actin":
        specific_headers = [
            "M Background", "A Background", "A localization",
            "A Lumen", "A Lumen/Bg",
            "t_cortex", "ISM", "Gini_Index", "Radial_Kurtosis",
        ]

    else:
        # No proteins present (e.g. Empty condition).
        # We still have membrane background and all membrane-derived
        # quality metrics — these go into membrane_only_headers below.
        specific_headers = [
            "M Background",
            "t_cortex", "ISM", "Gini_Index", "Radial_Kurtosis",
        ]

    # Build the full column list:
    #   base columns + protein-specific columns + membrane-only columns
    # The membrane-only columns come LAST so they align with the values
    # appended at the end of format_result_row().
    column_headers = base_headers + specific_headers + membrane_only_headers

    output_csv_path = os.path.join(final_output_path, "Analysis_Results.csv")

    with open(output_csv_path, "w", newline='') as output_file:
        df = csv.DictWriter(output_file, delimiter=',', fieldnames=column_headers)
        df.writeheader()

    return output_csv_path


# =============================================================================
# COLORMAPS
# =============================================================================

def color_maps():
    """
    Creates and registers the custom colormaps used for multi-channel display.
    """
    if "cmap_cyan" not in plt.colormaps():
        cmap_cyan = LinearSegmentedColormap.from_list(
            "cmap_cyan", list(zip([0.0, 1.0], ["black", "cyan"])))
        plt.colormaps.register(cmap=cmap_cyan)

    if "cmap_magenta" not in plt.colormaps():
        cmap_magenta = LinearSegmentedColormap.from_list(
            "cmap_magenta", list(zip([0.0, 1.0], ["black", "magenta"])))
        plt.colormaps.register(cmap=cmap_magenta)

    if "cmap_magenta_enhanced" not in plt.colormaps():
        cmap_magenta_enhanced = LinearSegmentedColormap.from_list(
            "cmap_magenta_enhanced",
            list(zip([0.0, 0.2, 1.0], ["black", "magenta", "magenta"])))
        plt.colormaps.register(cmap=cmap_magenta_enhanced)

    if "cmap_yellow" not in plt.colormaps():
        cmap_yellow = LinearSegmentedColormap.from_list(
            "cmap_yellow", list(zip([0.0, 1.0], ["black", "yellow"])))
        plt.colormaps.register(cmap=cmap_yellow)


# =============================================================================
# OVERVIEW PLOT
# =============================================================================

def detected_centres(to_plot, num_vesicles, ch_membrane, all_xc, all_yc,
                     final_output_path, exp_info):
    """
    Saves an overview image annotated with all detected vesicle centres.

    FIX: plt.close() was missing in the original — this caused a memory leak
    when processing many image sets in sequence, because every figure stayed
    open in memory.
    """
    if to_plot:
        plt.figure(dpi=125)
        plt.title('Total number of detected vesicles: ' + str(num_vesicles))
        plt.axis('off')
        plt.imshow(ch_membrane, cmap="cmap_cyan")
        plt.scatter(all_xc, all_yc, color='red', s=1, marker="o")
        for i in range(num_vesicles):
            plt.annotate(i + 1, (all_xc[i] + 5, all_yc[i] - 5), c='red')

        os.makedirs(final_output_path, exist_ok=True)
        plt.savefig(os.path.join(final_output_path, "Overview_Detected_Centres.png"))
        plt.close()   # FIX: was missing — prevented memory leak


# =============================================================================
# IMAGE CROPPING
# =============================================================================

def zoom_in_vesicle(channel, size_side, image_dim, ves_coordinates):
    """
    Crops a square region centred on a single vesicle.
    Pads with zeros if the vesicle is near the image edge.
    """
    xc   = ves_coordinates[1]
    yc   = ves_coordinates[2]
    radius = ves_coordinates[3]

    vesicle_box_side = int(size_side * radius)
    expected_size    = 2 * vesicle_box_side

    y_start_ideal = int(yc - vesicle_box_side)
    y_end_ideal   = int(yc + vesicle_box_side)
    x_start_ideal = int(xc - vesicle_box_side)
    x_end_ideal   = int(xc + vesicle_box_side)

    y_start_valid = max(0, y_start_ideal)
    y_end_valid   = min(image_dim[1], y_end_ideal)
    x_start_valid = max(0, x_start_ideal)
    x_end_valid   = min(image_dim[0], x_end_ideal)

    vesicle_slice = channel[y_start_valid:y_end_valid, x_start_valid:x_end_valid]

    if vesicle_slice.shape != (expected_size, expected_size):
        padded_box = np.zeros((expected_size, expected_size), dtype=channel.dtype)
        paste_y = max(0, -y_start_ideal)
        paste_x = max(0, -x_start_ideal)
        h_slice, w_slice = vesicle_slice.shape
        padded_box[paste_y:paste_y + h_slice, paste_x:paste_x + w_slice] = vesicle_slice
        return padded_box

    return vesicle_slice


# =============================================================================
# THRESHOLD
# =============================================================================

def define_threshold(channel, size_side, image_dim, ves_coordinates,
                     analysis_config):
    """
    Computes the intensity threshold for background masking for one vesicle,
    using that vesicle's own local image region.

    CHANGE: The old signature was define_threshold(user_choice, channel, ...).
    The `user_choice` bool is now read from analysis_config instead, making
    the function consistent with how all other settings are passed.

    CHANGE: This is now called once per vesicle inside process_single_vesicle
    rather than once globally in main.py. This means each vesicle gets a
    threshold calibrated to its own brightness, rather than reusing the
    threshold from the first vesicle in the image.

    PARAMETERS:
    -----------
    channel         : 2D array — the membrane (C1) image
    size_side       : float    — zoom box size in units of vesicle radius
    image_dim       : array    — (X_size, Y_size) of the full image
    ves_coordinates : array    — [id, xc, yc, radius]
    analysis_config : dict     — must contain 'threshold_method_manual' (bool)
                                 False → automatic Li threshold (parallel-safe)
                                 True  → interactive, requires n_jobs=1
    """
    vesicle_box_channel = zoom_in_vesicle(channel, size_side, image_dim,
                                          ves_coordinates)

    user_choice = analysis_config.get('threshold_method_manual', False)

    if user_choice:
        fig, ax = try_all_threshold(vesicle_box_channel, figsize=(5, 10),
                                    verbose=False)
        threshold_method = input("Choose threshold method to follow: ")

        methods = {
            'Isodata' : threshold_isodata,
            'Li'      : threshold_li,
            'Mean'    : threshold_mean,
            'Minimum' : threshold_minimum,
            'Otsu'    : threshold_otsu,
            'Triangle': threshold_triangle,
            'Yen'     : threshold_yen,
        }
        threshold = methods.get(threshold_method, threshold_li)(vesicle_box_channel)
    else:
        threshold = threshold_li(vesicle_box_channel)

    return threshold


# =============================================================================
# CENTRE REFINEMENT
# =============================================================================

def refine_guv_center(image, center_guess, radius_estimate, search_box_factor=1.5):
    """
    Refines the vesicle centre using edge gradients and the Hough transform.
    """
    xc, yc = center_guess
    box_half_width = int(radius_estimate * search_box_factor)

    x_start = int(max(0, xc - box_half_width))
    x_end   = int(min(image.shape[1], xc + box_half_width))
    y_start = int(max(0, yc - box_half_width))
    y_end   = int(min(image.shape[0], yc + box_half_width))

    if x_start >= x_end or y_start >= y_end:
        return center_guess

    roi = image[y_start:y_end, x_start:x_end]
    roi_center_x = xc - x_start
    roi_center_y = yc - y_start

    if roi.size == 0:
        return center_guess

    roi_norm   = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe      = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    roi_eq     = clahe.apply(roi_norm)
    roi_smooth = cv2.GaussianBlur(roi_eq, (5, 5), 0)

    edges = cv2.Canny(roi_smooth, 30, 100)
    gx    = cv2.Sobel(roi_smooth, cv2.CV_64F, 1, 0, ksize=5)
    gy    = cv2.Sobel(roi_smooth, cv2.CV_64F, 0, 1, ksize=5)

    y_edge, x_edge = np.where(edges > 0)
    rx = x_edge - roi_center_x
    ry = y_edge - roi_center_y
    r_norm = np.sqrt(rx**2 + ry**2)
    r_norm[r_norm == 0] = 1

    g_x_val = gx[y_edge, x_edge]
    g_y_val = gy[y_edge, x_edge]
    g_norm  = np.sqrt(g_x_val**2 + g_y_val**2)
    g_norm[g_norm == 0] = 1

    dot_product   = np.abs((g_x_val / g_norm) * (rx / r_norm) +
                           (g_y_val / g_norm) * (ry / r_norm))
    valid_indices = dot_product > 0.6

    clean_edges = np.zeros_like(edges)
    clean_edges[y_edge[valid_indices], x_edge[valid_indices]] = 255

    h, w = roi.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_guess = np.sqrt((X - roi_center_x)**2 + (Y - roi_center_y)**2)
    mask = ((dist_from_guess > radius_estimate * 0.75) &
            (dist_from_guess < radius_estimate * 1.25))
    clean_edges[~mask] = 0

    if np.sum(clean_edges) < 10:
        return center_guess

    hough_radii = np.arange(int(radius_estimate * 0.8), int(radius_estimate * 1.2), 2)
    if len(hough_radii) == 0:
        return center_guess

    hough_res = hough_circle(clean_edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)

    if len(cx) == 0:
        return center_guess

    return (x_start + cx[0], y_start + cy[0])


# =============================================================================
# LINEAR PROFILES
# =============================================================================

def linear_profiles(channels_data, ves_coordinates, image_dim, parameters_profiles):
    """
    Calculates the set of radial intensity profiles for a single vesicle.
    """
    num_channels = channels_data.shape[2]
    xc     = ves_coordinates[1]
    yc     = ves_coordinates[2]
    radius = ves_coordinates[3]

    xc_refined, yc_refined = refine_guv_center(
        channels_data[:, :, 0], (xc, yc), radius)

    ves_coordinates[1] = xc_refined
    ves_coordinates[2] = yc_refined

    num_angles    = int(parameters_profiles[0])
    length_excess = parameters_profiles[1]
    dr            = parameters_profiles[2]

    profile_radius_limit = length_excess * radius

    theta        = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    along_radius = np.arange(0, int(profile_radius_limit), dr)
    profile_radius = len(along_radius)

    intensity_profiles = np.zeros((profile_radius, num_angles, num_channels))

    death_mark = False
    if (xc_refined + profile_radius_limit >= image_dim[0] or
            xc_refined - profile_radius_limit < 0 or
            yc_refined + profile_radius_limit >= image_dim[1] or
            yc_refined - profile_radius_limit < 0):
        death_mark = True
        return intensity_profiles, along_radius, theta, death_mark

    x_coords = xc_refined + along_radius[np.newaxis, :] * np.cos(theta[:, np.newaxis])
    y_coords = yc_refined + along_radius[np.newaxis, :] * np.sin(theta[:, np.newaxis])

    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=0)

    for k in range(num_channels):
        channel = channels_data[:, :, k]
        profiles_flat = map_coordinates(channel, coords, order=1,
                                        mode='constant', cval=0.0)
        intensity_profiles[:, :, k] = profiles_flat.reshape(
            num_angles, profile_radius).T

    return intensity_profiles, along_radius, theta, death_mark


# =============================================================================
# MASK FILLING
# =============================================================================

def fill_in_mask(mask_memb):
    """
    Fills the interior of the vesicle contour mask.
    """
    mask_memb_fill = nd.binary_fill_holes(mask_memb).astype(int)

    border_cuts_left = np.where(mask_memb[:, 0] == 1)[0]
    if len(border_cuts_left) > 2:
        assistant_left = np.zeros((len(mask_memb[:, 0]), 1))
        assistant_left[border_cuts_left[0]:border_cuts_left[-1]] = 1
        mask_left = np.hstack((assistant_left, mask_memb_fill))
        mask_memb_fill = nd.binary_fill_holes(mask_left).astype(int)[:, 1:]

    border_cuts_right = np.where(mask_memb[:, -1] == 1)[0]
    if len(border_cuts_right) > 2:
        assistant_right = np.zeros((len(mask_memb[:, -1]), 1))
        assistant_right[border_cuts_right[0]:border_cuts_right[-1]] = 1
        mask_right = np.hstack((mask_memb_fill, assistant_right))
        mask_memb_fill = nd.binary_fill_holes(mask_right).astype(int)[:, :-1]

    border_cuts_up = np.where(mask_memb[0, :] == 1)[0]
    if len(border_cuts_up) > 2:
        assistant_up = np.zeros((1, len(mask_memb[0, :])))
        assistant_up[0, border_cuts_up[0]:border_cuts_up[-1]] = 1
        mask_up = np.vstack((assistant_up, mask_memb_fill))
        mask_memb_fill = nd.binary_fill_holes(mask_up).astype(int)[1:, :]

    border_cuts_down = np.where(mask_memb[-1, :] == 1)[0]
    if len(border_cuts_down) > 2:
        assistant_down = np.zeros((1, len(mask_memb[0, :])))
        assistant_down[0, border_cuts_down[0]:border_cuts_down[-1]] = 1
        mask_down = np.vstack((mask_memb_fill, assistant_down))
        mask_memb_fill = nd.binary_fill_holes(mask_down).astype(int)[:-1, :]

    return mask_memb_fill


# =============================================================================
# BACKGROUND NOISE
# =============================================================================

def background_noise(plot_mask, channels_data, proteins_present, size_mask,
                     image_dim, ves_coordinates, threshold, final_output_path,
                     exp_info):
    """
    Estimates background signal outside the vesicle membrane.
    """
    num_channels = channels_data.shape[2]
    size_box = int(size_mask * ves_coordinates[3])

    vesicle_box = np.zeros((2 * size_box, 2 * size_box, num_channels))
    for i in range(num_channels):
        vesicle_box[:, :, i] = zoom_in_vesicle(
            channels_data[:, :, i], size_mask, image_dim, ves_coordinates)

    mask_memb      = vesicle_box[:, :, 0] > threshold
    mask_memb_fill = fill_in_mask(mask_memb)

    background = np.zeros((1, num_channels))
    for i in range(num_channels):
        background[:, i] = np.mean(
            np.ma.array(vesicle_box[:, :, i], mask=mask_memb_fill))

    if plot_mask:
        protein_status = plot_format(proteins_present)
        if protein_status == "both_proteins":
            show_mask_both(int(ves_coordinates[0]), vesicle_box, mask_memb_fill,
                           background, final_output_path, exp_info)
        elif protein_status == "only_septin":
            show_mask_only_sept(int(ves_coordinates[0]), vesicle_box, mask_memb_fill,
                                background, final_output_path, exp_info)
        elif protein_status == "only_actin":
            show_mask_only_actin(int(ves_coordinates[0]), vesicle_box, mask_memb_fill,
                                 background, final_output_path, exp_info)

    return background


# =============================================================================
# FORMAT HELPERS
# =============================================================================

def plot_format(proteins_present):
    """
    Returns a string describing which protein channels are present.
    """
    if proteins_present[0] and proteins_present[1]: return "both_proteins"
    if proteins_present[0]: return "only_septin"
    if proteins_present[1]: return "only_actin"
    return "unknown"


# =============================================================================
# MASK PLOTS
# =============================================================================

def show_mask_both(vesicle_id, vesicle_box, mask_memb_fill, background,
                   path_to_output, exp_info):
    """Saves a background mask visualisation (both proteins present)."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(13, 4), dpi=125)
    fig.suptitle(
        f'Vesicle {vesicle_id}  Background:  '
        f'M:{round(background[0,0],3)}, '
        f'S:{round(background[0,1],3)}, '
        f'A:{round(background[0,2],3)}')

    ax1.imshow(vesicle_box[:, :, 0], cmap="cmap_cyan");          ax1.set_title('Membrane'); ax1.set_axis_off()
    ax2.imshow(vesicle_box[:, :, 1], cmap="cmap_magenta_enhanced"); ax2.set_title('Septin');   ax2.set_axis_off()
    ax3.imshow(vesicle_box[:, :, 2], cmap="cmap_yellow");         ax3.set_title('Actin');    ax3.set_axis_off()
    ax4.imshow(mask_memb_fill, cmap="binary_r");                  ax4.set_title('Mask');     ax4.set_axis_off()

    os.makedirs(path_to_output, exist_ok=True)
    fig.savefig(os.path.join(path_to_output, f"Vesicle_{vesicle_id}_Channels_and_Mask.png"))
    plt.close(fig)


def show_mask_only_sept(vesicle_id, vesicle_box, mask_memb_fill, background,
                        path_to_output, exp_info):
    """Saves a background mask visualisation (Septin only)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5), dpi=125)
    fig.suptitle(
        f'Vesicle {vesicle_id}  Background:  '
        f'M:{round(background[0,0],3)}, S:{round(background[0,1],3)}')

    ax1.imshow(vesicle_box[:, :, 0], cmap="cmap_cyan");             ax1.set_title('Membrane'); ax1.set_axis_off()
    ax2.imshow(vesicle_box[:, :, 1], cmap="cmap_magenta_enhanced"); ax2.set_title('Septin');   ax2.set_axis_off()
    ax3.imshow(mask_memb_fill, cmap="binary_r");                    ax3.set_title('Mask');     ax3.set_axis_off()

    os.makedirs(path_to_output, exist_ok=True)
    fig.savefig(os.path.join(path_to_output, f"Vesicle_{vesicle_id}-Channels_and_Mask.png"))
    plt.close(fig)


def show_mask_only_actin(vesicle_id, vesicle_box, mask_memb_fill, background,
                         path_to_output, exp_info):
    """Saves a background mask visualisation (Actin only)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5), dpi=125)
    fig.suptitle(
        f'Vesicle {vesicle_id}  Background:  '
        f'M:{round(background[0,0],3)}, A:{round(background[0,1],3)}')

    ax1.imshow(vesicle_box[:, :, 0], cmap="cmap_cyan");   ax1.set_title('Membrane'); ax1.set_axis_off()
    ax2.imshow(vesicle_box[:, :, 1], cmap="cmap_yellow"); ax2.set_title('Actin');    ax2.set_axis_off()
    ax3.imshow(mask_memb_fill, cmap="binary_r");           ax3.set_title('Mask');     ax3.set_axis_off()

    os.makedirs(path_to_output, exist_ok=True)
    fig.savefig(os.path.join(path_to_output, f"Vesicle_{vesicle_id}-Channels_and_Mask.png"))
    plt.close(fig)


# =============================================================================
# BACKGROUND CORRECTION
# =============================================================================

def background_correction(num_channels, intensity_profiles, background):
    """
    Subtracts background from every channel. Clips at zero (no negative values).
    """
    intensity_profiles_corrected = np.ones_like(intensity_profiles)
    for i in range(num_channels):
        intensity_profiles_corrected[:, :, i] = np.clip(
            intensity_profiles[:, :, i] - background[:, i], 0, None)
    return intensity_profiles_corrected


# =============================================================================
# RADIAL PROFILE
# =============================================================================

def radial_profile(num_channels, intensity_profiles):
    """
    Collapses the angular dimension by averaging, giving one radial curve
    per channel.
    """
    radial_profiles = np.ones_like(intensity_profiles[:, 0, :])
    for i in range(num_channels):
        radial_profiles[:, i] = np.average(intensity_profiles[:, :, i], axis=1)
    return radial_profiles


# =============================================================================
# MEMBRANE DETECTION
# =============================================================================

def membrane_detection(radial_profile_memb, radius):
    """
    Detects the inner and outer borders of the membrane ring in the
    averaged radial profile.
    Returns peak_index for use in refined radius calculation.
    """
    comment    = []
    death_mark = False
    peak_index = 0

    peaks, properties = find_peaks(
        radial_profile_memb,
        height=np.max(radial_profile_memb) * 0.1,
        distance=5,
        prominence=np.max(radial_profile_memb) * 0.05)

    if not np.any(peaks):
        return 0, 0, 0, ["no_memb_peak"], True

    if len(peaks) == 1:
        chosen_peak = 0
    else:
        peak_scores = []
        for i, peak_pos in enumerate(peaks):
            dist_score   = 1.0 / (1.0 + abs(peak_pos - radius) / radius)
            height_score = radial_profile_memb[peak_pos] / np.max(radial_profile_memb)
            peak_scores.append(dist_score * 3.0 + height_score * 2.0)
        chosen_peak = np.argmax(peak_scores)

    peak_index = peaks[chosen_peak]

    try:
        width_results  = peak_widths(radial_profile_memb, [peak_index], rel_height=0.75)
        left_idx       = width_results[2][0]
        right_idx      = width_results[3][0]
        padding        = 3
        index_border_in  = max(0, int(np.floor(left_idx  - padding)))
        index_border_out = min(len(radial_profile_memb) - 1,
                               int(np.ceil(right_idx + padding)))
    except Exception:
        index_border_in  = max(0, peaks[chosen_peak] - 8)
        index_border_out = min(len(radial_profile_memb) - 1, peaks[chosen_peak] + 8)
        comment.append("width_calc_failed")

    if index_border_out - index_border_in > radius / 2:
        comment.append("wide_peak")

    return index_border_in, index_border_out, peak_index, comment, death_mark


# =============================================================================
# MEMBRANE QUALITY CHECK
# =============================================================================

def check_membrane_quality(angular_profiles, radial_profiles, radius):
    """
    Flags potential membrane quality issues without rejecting the vesicle.

    FIX: The original code calculated the CV twice inside nested if-blocks
    that checked the same condition. The inner duplicate block has been removed.
    """
    comments = []
    membrane_angular = angular_profiles[:, 0]

    # --- CV of membrane brightness around the ring ---
    # FIX: Previously computed twice inside two identical if-conditions.
    # Now computed once, used once.
    if np.mean(membrane_angular) > 0:
        cv_membrane = np.std(membrane_angular) / np.mean(membrane_angular)
        if   cv_membrane > 0.4: comments.append("very_irregular_membrane")
        elif cv_membrane > 0.3: comments.append("irregular_membrane")

    # CHECK 1: Sharp angular transitions
    if len(membrane_angular) > 10:
        membrane_diff  = np.abs(np.diff(membrane_angular))
        max_transition = np.max(membrane_diff)
        mean_signal    = np.mean(membrane_angular)
        if max_transition > mean_signal * 0.5:
            comments.append("sharp_membrane_transitions")

    # CHECK 2: Plateau-like flat regions
    if len(membrane_angular) > 20:
        rolling_std = []
        window_size = 20
        for i in range(len(membrane_angular) - window_size):
            rolling_std.append(np.std(membrane_angular[i:i + window_size]))
        if rolling_std:
            if np.min(rolling_std) < 5 and np.max(rolling_std) > 15:
                comments.append("membrane_plateau_regions")

    # CHECK 3: Protein clustering
    if angular_profiles.shape[1] > 1:
        for protein_idx in range(1, angular_profiles.shape[1]):
            protein_angular = angular_profiles[:, protein_idx]
            if np.mean(protein_angular) > 5:
                protein_cv = np.std(protein_angular) / np.mean(protein_angular)
                if   protein_cv > 0.8: comments.append("high_protein_clustering")
                elif protein_cv > 0.6: comments.append("moderate_protein_clustering")

    # CHECK 4: Asymmetric membrane (front vs back)
    if len(membrane_angular) >= 180:
        half      = len(membrane_angular) // 2
        first_half  = membrane_angular[:half]
        second_half = membrane_angular[half:half * 2]
        if np.mean(first_half) > 0 and np.mean(second_half) > 0:
            asymmetry_ratio = (
                abs(np.mean(first_half) - np.mean(second_half)) /
                max(np.mean(first_half), np.mean(second_half)))
            if asymmetry_ratio > 0.3:
                comments.append("asymmetric_membrane")

    # CHECK 5: Spikes and valleys
    if len(membrane_angular) > 0:
        membrane_median = np.median(membrane_angular)
        if np.max(membrane_angular) > membrane_median * 3:
            comments.append("membrane_spikes")
        if membrane_median > 0 and np.min(membrane_angular) < membrane_median * 0.3:
            comments.append("membrane_valleys")

    # CHECK 6: Overall signal strength
    mean_membrane = np.mean(membrane_angular)
    if   mean_membrane < 10: comments.append("weak_membrane_signal")
    elif mean_membrane < 20: comments.append("low_membrane_signal")

    return comments


# =============================================================================
# ANGULAR PROFILE
# =============================================================================

def angular_profile(num_channels, intensity_profiles, index_border_in,
                    index_border_out, pixels_to_remove):
    """
    Extracts the angular (membrane-ring) profile for each channel.

    Uses np.max instead of np.average so that sparse protein patches
    are not diluted by low-signal radial positions within the membrane
    border window.

    Note on pixels_to_remove offset:
        intensity_profiles here is the FULL (unsliced) corrected array.
        index_border_in / out are indices into the SLICED radial profile
        (which had pixels_to_remove rows removed from the front).
        Adding pixels_to_remove back converts these indices to the
        correct rows in the full array.
    """
    angular_profiles = np.ones_like(intensity_profiles[0, :, :])

    for i in range(num_channels):
        segment = intensity_profiles[
            pixels_to_remove + index_border_in:
            pixels_to_remove + index_border_out, :, i]
        angular_profiles[:, i] = np.max(segment, axis=0)

    return angular_profiles


# =============================================================================
# LOCALIZATION
# =============================================================================

def localization(num_channels, angular_profiles, radial_profiles,
                 index_border_in, index_border_out, radius,
                 size_central_area, pixels_to_remove):
    """
    Quantifies protein localization on the membrane relative to the lumen.
    """
    comment    = []
    num_proteins = num_channels - 1
    index_centre = int(size_central_area * (radius - pixels_to_remove))
    localization_res    = np.zeros(num_proteins)
    lumen_intensity_res = np.zeros(num_proteins)

    NOISE_FLOOR = 4.0
    MIN_Z_SCORE = 3.0

    for i in range(num_proteins):
        prot_idx    = i + 1
        memb_angular = angular_profiles[:, prot_idx]
        median_memb  = np.median(memb_angular)

        lumen_radial = radial_profiles[0:index_centre, prot_idx]
        mu_lumen     = np.median(lumen_radial)   # Median is more robust than mean against
                                                 # bright outlier pixels (e.g. a stray actin
                                                 # filament passing through the lumen region).
        sigma_lumen  = np.std(lumen_radial)
        lumen_intensity_res[i] = mu_lumen

        if sigma_lumen < 0.001:
            sigma_lumen = 0.1

        if median_memb < NOISE_FLOOR:
            localization_res[i] = 0.0
            continue

        b_in  = max(0, int(index_border_in))
        b_out = min(len(radial_profiles), int(index_border_out))

        if b_in < b_out:
            peak_intensity = np.max(radial_profiles[b_in:b_out, prot_idx])
        else:
            peak_intensity = median_memb

        z_score = (peak_intensity - mu_lumen) / sigma_lumen

        if z_score < MIN_Z_SCORE:
            localization_res[i] = 0.0
            comment.append(f"low_SNR_z{z_score:.1f}")
        else:
            if median_memb > 0.001:
                score = (median_memb - mu_lumen) / median_memb
                localization_res[i] = np.clip(score, 0, None)
            else:
                localization_res[i] = 0.0

    return localization_res, lumen_intensity_res, comment


# =============================================================================
# ACTIN STRUCTURE METRICS
# =============================================================================

def analyze_actin_structure(radial_profiles, angular_profiles,
                             localization_score, protein_channel_index,
                             px_size=1.0, border_in=0, border_out=0):
    """
    Calculates advanced physical descriptors for actin:
      1. Gini coefficient     (spatial clustering)
      2. ISM                  (integrated surface mass)
      3. Radial kurtosis      (sharpness of the cortical ring)
      4. t_cortex / FWHM      (cortical thickness)
    """
    t_cortex = ism = gini = rad_kurtosis = 0.0

    if localization_score <= 0.0:
        return t_cortex, ism, gini, rad_kurtosis

    actin_angular = angular_profiles[:, protein_channel_index]
    actin_radial  = radial_profiles[:, protein_channel_index]

    # 1. Gini coefficient
    noise_floor             = np.percentile(actin_angular, 10)
    actin_angular_corrected = np.clip(actin_angular - noise_floor, 0, None)
    sorted_angular = np.sort(actin_angular_corrected)
    n = len(sorted_angular)
    if np.mean(sorted_angular) > 0:
        index = np.arange(1, n + 1)
        gini  = (np.sum((2 * index - n - 1) * sorted_angular) /
                 (n * np.sum(sorted_angular)))

    # 2. Integrated Surface Mass (ISM)
    if border_out > border_in:
        r_indices  = np.arange(border_in, border_out)
        ring_areas = 2 * np.pi * r_indices * px_size
        limit = min(len(actin_radial), len(ring_areas) + border_in)
        ism   = np.sum(actin_radial[border_in:limit] * ring_areas[:limit - border_in])

    # 3. Radial kurtosis
    if np.max(actin_radial) > 0:
        slice_start    = max(0, border_in - 5)
        slice_end      = min(len(actin_radial), border_out + 5)
        cortex_segment = actin_radial[slice_start:slice_end]
        if len(cortex_segment) > 3:
            rad_kurtosis = kurtosis(cortex_segment)

    # 4. Cortical thickness (FWHM)
    if localization_score >= 0.2:
        peaks, _ = find_peaks(actin_radial, height=np.max(actin_radial) * 0.5)
        if len(peaks) > 0:
            idx = np.argmax(actin_radial[peaks])
            widths, _, _, _ = peak_widths(actin_radial, peaks, rel_height=0.5)
            t_cortex = widths[idx] * px_size

    return t_cortex, ism, gini, rad_kurtosis


# =============================================================================
# DEBUG OVERLAY PLOT
# =============================================================================

def plot_debug_overlay(ves_coordinates, radius, index_border_in,
                       index_border_out, image_dim, channels_data,
                       along_radius, theta, final_output_path):
    """
    Saves a debug image showing radial spokes and detected membrane borders.
    """
    membrane_channel = channels_data[:, :, 0]
    vesicle_img      = zoom_in_vesicle(membrane_channel, 1.5, image_dim,
                                       ves_coordinates)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(vesicle_img, cmap='gray')
    ax.set_title(f"Debug: Vesicle {int(ves_coordinates[0])}")

    center_img = vesicle_img.shape[0] // 2

    for angle in theta[::15]:
        r_max = along_radius[-1]
        x_end = center_img + r_max * np.cos(angle)
        y_end = center_img + r_max * np.sin(angle)
        ax.plot([center_img, x_end], [center_img, y_end],
                color='yellow', alpha=0.3, lw=0.5)

    radius_in  = along_radius[index_border_in]
    radius_out = along_radius[index_border_out]

    circ_in  = plt.Circle((center_img, center_img), radius_in,
                           color='red',    fill=False, lw=1.5, label='Inner Border')
    circ_out = plt.Circle((center_img, center_img), radius_out,
                           color='orange', fill=False, lw=1.5, label='Outer Border')
    ax.add_patch(circ_in)
    ax.add_patch(circ_out)
    ax.legend(loc='upper right', fontsize='small')
    ax.axis('off')

    os.makedirs(final_output_path, exist_ok=True)
    fig.savefig(os.path.join(
        final_output_path,
        f"Vesicle_{int(ves_coordinates[0])}_DEBUG.png"))
    plt.close(fig)


# =============================================================================
# INTENSITY PROFILE PLOTS
# =============================================================================

def plot_intensity_profiles(plot_int_profiles, proteins_present, parameters_sizes,
                             channels_data, ves_coordinates, image_dim,
                             along_radius, theta, radial_profiles, angular_profiles,
                             index_border_in, index_border_out, background,
                             localization, path_to_output, exp_info, pixel_size):
    """
    Dispatches to the correct per-protein-combination plotting function.
    """
    if not plot_int_profiles:
        return

    num_channels = channels_data.shape[2]
    size_box = int(parameters_sizes[1] * ves_coordinates[3])

    image_box = np.zeros((2 * size_box, 2 * size_box, num_channels))
    for i in range(num_channels):
        image_box[:, :, i] = zoom_in_vesicle(
            channels_data[:, :, i], parameters_sizes[1],
            image_dim, ves_coordinates)

    protein_status = plot_format(proteins_present)

    if protein_status == "both_proteins":
        plot_int_prof_both(
            int(ves_coordinates[0]), ves_coordinates[3], parameters_sizes[2],
            along_radius, theta, radial_profiles, angular_profiles,
            index_border_in, index_border_out, image_box, background,
            localization, path_to_output, exp_info, pixel_size)

    elif protein_status == "only_septin":
        plot_int_prof_only_septin(
            int(ves_coordinates[0]), ves_coordinates[3], parameters_sizes[2],
            along_radius, theta, radial_profiles, angular_profiles,
            index_border_in, index_border_out, image_box, background,
            localization, path_to_output, exp_info, pixel_size)

    elif protein_status == "only_actin":
        plot_int_prof_only_actin(
            int(ves_coordinates[0]), ves_coordinates[3], parameters_sizes[2],
            along_radius, theta, radial_profiles, angular_profiles,
            index_border_in, index_border_out, image_box, background,
            localization, path_to_output, exp_info, pixel_size)


def plot_int_prof_both(vesicle_id, radius, size_central_area, along_radius,
                       theta, radial_profiles, angular_profiles,
                       index_border_in, index_border_out, image_box,
                       background, localization, path_to_output, exp_info,
                       pixel_size):
    """Saves radial and angular intensity profile plots (both proteins)."""
    radius_um = along_radius * pixel_size

    fig, axes = plt.subplot_mosaic("AB;AC;AD;AE", width_ratios=[3, 1], dpi=125)
    axes["A"].set_title(f'Intensity radial profile of Vesicle {vesicle_id}')
    axes["A"].plot(radius_um, radial_profiles[:, 0], c='cyan',    label='membrane')
    axes["A"].plot(radius_um, radial_profiles[:, 1], c='magenta', label='septin')
    axes["A"].plot(radius_um, radial_profiles[:, 2], c='gold',    label='actin')

    actin_prof = radial_profiles[:, 2]
    peaks, _   = find_peaks(actin_prof, height=np.max(actin_prof) * 0.5)
    if len(peaks) > 0:
        widths, width_heights, left_ips, right_ips = peak_widths(
            actin_prof, peaks, rel_height=0.5)
        idx = np.argmax(actin_prof[peaks])
        axes["A"].hlines(width_heights[idx],
                         (along_radius[0] + left_ips[idx])  * pixel_size,
                         (along_radius[0] + right_ips[idx]) * pixel_size,
                         color='blue', lw=2, label='Actin FWHM')

    axes["A"].axvline(x=along_radius[index_border_in]  * pixel_size, c='red',    ls="--", label='border in')
    axes["A"].axvline(x=along_radius[index_border_out] * pixel_size, c='orange', ls="--", label='border out')
    axes["A"].set_xlabel('radius (µm)')
    axes["A"].set_ylabel('I (a.u.)')
    axes["A"].legend(loc="upper left", fontsize=8)

    axes["B"].imshow(image_box[:, :, 0], cmap="cmap_cyan");             axes["B"].set_axis_off()
    axes["C"].imshow(image_box[:, :, 1], cmap="cmap_magenta_enhanced"); axes["C"].set_axis_off()
    axes["D"].imshow(image_box[:, :, 2], cmap="cmap_yellow");           axes["D"].set_axis_off()

    axes["E"].text(-0.2,  0.7, f"Background M:  {round(background[0,0],4)}", size=10)
    axes["E"].text(-0.2,  0.5, f"Background S:  {round(background[0,1],4)}", size=10)
    axes["E"].text(-0.2,  0.3, f"Background A:  {round(background[0,2],4)}", size=10)
    axes["E"].text(-0.2,  0.0, f"Localization S: {round(localization[0],3)}", size=10)
    axes["E"].text(-0.2, -0.2, f"Localization A: {round(localization[1],3)}", size=10)
    axes["E"].set_axis_off()

    os.makedirs(path_to_output, exist_ok=True)
    fig.savefig(os.path.join(path_to_output, f"Vesicle_{vesicle_id}-Int_prof_Radial.png"))

    fig2, axes2 = plt.subplot_mosaic("AB;AC;AD", width_ratios=[3, 1], dpi=125)
    axes2["A"].set_title(f'Intensity angular profile of Vesicle {vesicle_id}')
    axes2["A"].plot(theta * 180 / np.pi, angular_profiles[:, 0], c='cyan',    label='membrane')
    axes2["A"].plot(theta * 180 / np.pi, angular_profiles[:, 1], c='magenta', label='septin')
    axes2["A"].plot(theta * 180 / np.pi, angular_profiles[:, 2], c='gold',    label='actin')
    axes2["A"].legend(loc="upper right")
    axes2["A"].set(xlabel='θ (deg)', ylabel='I (a.u.)')

    axes2["B"].imshow(image_box[:, :, 0], cmap="cmap_cyan");             axes2["B"].set_axis_off()
    axes2["C"].imshow(image_box[:, :, 1], cmap="cmap_magenta_enhanced"); axes2["C"].set_axis_off()
    axes2["D"].imshow(image_box[:, :, 2], cmap="cmap_yellow");           axes2["D"].set_axis_off()

    fig2.savefig(os.path.join(path_to_output, f"Vesicle_{vesicle_id}-Int_prof_Angular.png"))
    plt.close('all')


def plot_int_prof_only_septin(vesicle_id, radius, size_central_area, along_radius,
                               theta, radial_profiles, angular_profiles,
                               index_border_in, index_border_out, image_box,
                               background, localization, path_to_output, exp_info,
                               pixel_size):
    """Saves radial and angular intensity profile plots (Septin only)."""
    radius_um = along_radius * pixel_size

    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios=[6, 2], dpi=125)
    axes["A"].set_title(f'Intensity radial profile of Vesicle {vesicle_id}')
    axes["A"].plot(radius_um[2:], radial_profiles[2:, 0] / np.max(radial_profiles[2:, 0]),
                   c='cyan',    label='Membrane channel')
    axes["A"].plot(radius_um[2:], radial_profiles[2:, 1] / np.max(radial_profiles[2:, 1]),
                   c='magenta', label='Septin channel')
    axes["A"].set_xlabel('radius (µm)')
    axes["A"].set_ylabel('normalised intensity (a.u.)')

    axes["B"].imshow(image_box[:, :, 0], cmap="cmap_cyan");             axes["B"].set_axis_off()
    axes["C"].imshow(image_box[:, :, 1], cmap="cmap_magenta_enhanced"); axes["C"].set_axis_off()

    os.makedirs(path_to_output, exist_ok=True)
    fig.savefig(os.path.join(path_to_output, f"Vesicle_{vesicle_id}-Int_prof_Radial.png"))

    fig2, axes2 = plt.subplot_mosaic("AB;AC", width_ratios=[6, 2], dpi=125)
    axes2["A"].set_title(f'Intensity angular profile of Vesicle {vesicle_id}')
    axes2["A"].plot(theta * 180 / np.pi,
                    angular_profiles[:, 0] / max(angular_profiles[:, 0]),
                    c='cyan', label='Membrane channel')
    axes2["A"].plot(theta * 180 / np.pi,
                    angular_profiles[:, 1] / max(angular_profiles[:, 1]),
                    c='magenta', label='Septin channel')
    axes2["A"].legend(loc="upper right", fontsize=10)
    axes2["A"].set(xlabel='θ (deg)', ylabel='normalised intensity (a.u.)')

    axes2["B"].imshow(image_box[:, :, 0], cmap="cmap_cyan");             axes2["B"].set_axis_off()
    axes2["C"].imshow(image_box[:, :, 1], cmap="cmap_magenta_enhanced"); axes2["C"].set_axis_off()

    fig2.savefig(os.path.join(path_to_output, f"Vesicle_{vesicle_id}-Int_prof_Angular.png"))
    plt.close('all')


def plot_int_prof_only_actin(vesicle_id, radius, size_central_area, along_radius,
                              theta, radial_profiles, angular_profiles,
                              index_border_in, index_border_out, image_box,
                              background, localization, path_to_output, exp_info,
                              pixel_size):
    """Saves radial and angular intensity profile plots (Actin only)."""
    radius_um = along_radius * pixel_size

    fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios=[3, 1], dpi=125)
    axes["A"].set_title(f'Intensity radial profile of Vesicle {vesicle_id}')
    axes["A"].plot(radius_um, radial_profiles[:, 0], c='cyan', label='membrane')
    axes["A"].plot(radius_um, radial_profiles[:, 1], c='gold', label='actin')

    actin_prof = radial_profiles[:, 1]
    peaks, _   = find_peaks(actin_prof, height=np.max(actin_prof) * 0.5)
    if len(peaks) > 0:
        widths, width_heights, left_ips, right_ips = peak_widths(
            actin_prof, peaks, rel_height=0.5)
        idx = np.argmax(actin_prof[peaks])
        axes["A"].hlines(width_heights[idx],
                         (along_radius[0] + left_ips[idx])  * pixel_size,
                         (along_radius[0] + right_ips[idx]) * pixel_size,
                         color='blue', lw=2, label='Actin FWHM')

    axes["A"].axvline(x=along_radius[index_border_in]  * pixel_size, c='red',    ls="--", label='border in')
    axes["A"].axvline(x=along_radius[index_border_out] * pixel_size, c='orange', ls="--", label='border out')
    axes["A"].set_xlabel('radius (µm)')
    axes["A"].set_ylabel('I (a.u.)')
    axes["A"].legend(loc="upper left", fontsize=8)

    axes["B"].imshow(image_box[:, :, 0], cmap="cmap_cyan");   axes["B"].set_axis_off()
    axes["C"].imshow(image_box[:, :, 1], cmap="cmap_yellow"); axes["C"].set_axis_off()
    axes["D"].text(-0.2, 0.7, f"Background M:  {round(background[0,0],4)}", size=10)
    axes["D"].text(-0.2, 0.0, f"Localization A: {round(localization[0],3)}", size=10)
    axes["D"].set_axis_off()

    os.makedirs(path_to_output, exist_ok=True)
    fig.savefig(os.path.join(path_to_output, f"Vesicle_{vesicle_id}-Int_prof_Radial.png"))

    fig2, axes2 = plt.subplot_mosaic("AB;AC", width_ratios=[3, 1], dpi=125)
    axes2["A"].set_title(f'Intensity angular profile of Vesicle {vesicle_id}')
    axes2["A"].plot(theta * 180 / np.pi, angular_profiles[:, 0], c='cyan', label='membrane')
    axes2["A"].plot(theta * 180 / np.pi, angular_profiles[:, 1], c='gold', label='actin')
    axes2["A"].legend(loc="upper left")
    axes2["A"].set(xlabel='θ (deg)', ylabel='I (a.u.)')

    axes2["B"].imshow(image_box[:, :, 0], cmap="cmap_cyan");   axes2["B"].set_axis_off()
    axes2["C"].imshow(image_box[:, :, 1], cmap="cmap_yellow"); axes2["C"].set_axis_off()

    fig2.savefig(os.path.join(path_to_output, f"Vesicle_{vesicle_id}-Int_prof_Angular.png"))
    plt.close('all')


# =============================================================================
# SIZE DISTRIBUTION PLOT
# =============================================================================

def plot_size_distribution(refined_radii_um, path_to_output):
    """
    Saves a histogram of all refined vesicle radii (in µm).
    """
    if len(refined_radii_um) == 0:
        return

    plt.figure(figsize=(6, 4), dpi=125)
    plt.hist(refined_radii_um, bins=10, color='gray', edgecolor='black', alpha=0.7)
    plt.title(f"Size Distribution (N={len(refined_radii_um)})")
    plt.xlabel("Refined Radius (µm)")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)

    os.makedirs(path_to_output, exist_ok=True)
    plt.savefig(os.path.join(path_to_output, "Size_Distribution_Refined.png"))
    plt.close()


# =============================================================================
# CSV ROW ASSEMBLY
# =============================================================================

def format_result_row(exp_info, ves_coordinates, background, localization,
                      lumen_intensity, t_cortex, ism, gini, rad_kurtosis,
                      refined_radius_um, comment,
                      shape_deform_dict=None, shape_sector_data=None):
    """
    Assembles all per-vesicle results into a flat list ready to write as a
    CSV row. Does NOT write to file — that happens in main.py.
    """
    list_exp_info        = list(exp_info[:-1])
    list_ves_coordinates = list(ves_coordinates)
    list_background      = list(background[0, :])
    list_localization    = list(localization)
    list_lumen_data      = []
    num_proteins         = len(localization)

    for i in range(num_proteins):
        lumen_val = lumen_intensity[i]
        bg_val    = background[0, i + 1]
        ratio     = lumen_val / bg_val if bg_val > 0 else 0
        list_lumen_data.append(round(float(lumen_val), 3))
        list_lumen_data.append(round(float(ratio), 3))

    if t_cortex is not None:
        list_structure = [round(float(t_cortex), 3), round(float(ism), 3),
                          round(float(gini), 3), round(float(rad_kurtosis), 3)]
    else:
        list_structure = [np.nan, np.nan, np.nan, np.nan]

    radius_val = [round(float(refined_radius_um), 3)] if refined_radius_um else [0]

    if shape_deform_dict is not None and shape_sector_data is not None:
        list_shape_metrics = format_shape_metrics_for_csv(shape_deform_dict,
                                                          shape_sector_data)
    else:
        list_shape_metrics = [np.nan, "N/A", "N/A"]

    vesicle_row = (list_exp_info + list_ves_coordinates +
                   list_background + list_localization +
                   list_lumen_data + list_structure +
                   list_shape_metrics + radius_val + comment)

    return vesicle_row


# =============================================================================
# MAIN PER-VESICLE PIPELINE
# =============================================================================

def process_single_vesicle(ves_coordinates, channels_data, image_dim,
                            parameters_profiles, proteins_present, size_mask,
                            final_output_path, exp_info, parameters_sizes,
                            plot_int_profiles, plot_mask, pixel_size,
                            analysis_config):
    """
    Runs the complete analysis pipeline for one vesicle.

    Designed for parallel execution with joblib — returns a result row
    and refined radius instead of writing to disk directly.

    CHANGES FROM PREVIOUS VERSION:
    --------------------------------
    1. `threshold_membrane` parameter REMOVED.
       The threshold is computed here from this vesicle's own image region,
       so each vesicle gets its own calibrated threshold.

    2. `analysis_config` parameter ADDED.
       Replaces all hard-coded magic numbers (min_radius_um, pixels_to_remove,
       num_sectors, etc.) with values from main.py's ANALYSIS_CONFIG dict.

    3. `lumen_val` initialised at the TOP of the function.
       Previously it was only created inside GATE 1 (wide_peak/too_small).
       GATE 2 (death_mark_peak) then used lumen_val without it ever being
       created — a NameError crash. Now it always exists.

    4. `along_radius_full` is saved BEFORE the array is sliced.
       The sliced along_radius is used for membrane detection and everything
       downstream. The full version is passed to shape analysis, which must
       match the (also unsliced) intensity_profiles_corrected array.

    5. `analysis_config` is passed through to shape analysis functions so
       they use configurable thresholds instead of hard-coded values.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="some peaks have a width of 0")
    color_maps()

    pixels_to_remove = analysis_config.get('pixels_to_remove_centre', 2)
    min_radius_um    = analysis_config.get('min_vesicle_radius_um', 3.0)

    num_channels = channels_data.shape[2]
    comment      = []

    # Initialise lumen_val
    lumen_val = np.zeros(num_channels - 1)

    # ------------------------------------------------------------------
    # STEP 1 — Per-vesicle threshold
    # ------------------------------------------------------------------
    threshold_membrane = define_threshold(
        channels_data[:, :, 0],
        size_mask,
        image_dim,
        ves_coordinates,
        analysis_config
    )

    # ------------------------------------------------------------------
    # STEP 2 — Radial profiles
    # ------------------------------------------------------------------
    intensity_profiles, along_radius, theta, death_mark = linear_profiles(
        channels_data, ves_coordinates, image_dim, parameters_profiles)

    if death_mark:
        background       = np.zeros((1, num_channels))
        localization_val = np.zeros(num_channels - 1)
        comment = ["margins"]
        row = format_result_row(
            exp_info, ves_coordinates, background, localization_val,
            lumen_val, None, None, None, None, 0, comment,
            shape_deform_dict=None, shape_sector_data=None)
        return row, None

    # ------------------------------------------------------------------
    # STEP 3 — Background correction
    # ------------------------------------------------------------------
    background = background_noise(
        plot_mask, channels_data, proteins_present, size_mask, image_dim,
        ves_coordinates, threshold_membrane, final_output_path, exp_info)
    intensity_profiles_corrected = background_correction(
        num_channels, intensity_profiles, background)

    # ------------------------------------------------------------------
    # STEP 4 — Radial profile (angle-averaged)
    # ------------------------------------------------------------------
    radial_profiles = radial_profile(num_channels, intensity_profiles_corrected)

    # Save the FULL along_radius BEFORE slicing.
    # Shape analysis must receive an along_radius whose length matches
    # intensity_profiles_corrected (which is never sliced).
    # Everything AFTER this line uses the sliced version.
    along_radius_full = along_radius.copy()

    radial_profiles = radial_profiles[pixels_to_remove:, :]
    along_radius    = along_radius[pixels_to_remove:]

    # ------------------------------------------------------------------
    # STEP 5 — Membrane detection
    # ------------------------------------------------------------------
    (index_border_in, index_border_out, peak_index,
     comment_peak, death_mark_peak) = membrane_detection(
        radial_profiles[:, 0], ves_coordinates[3])

    if comment_peak:
        comment += comment_peak

    refined_radius_px = along_radius[peak_index]
    refined_radius_um = refined_radius_px * pixel_size

    plot_debug_overlay(
        ves_coordinates, ves_coordinates[3],
        index_border_in, index_border_out,
        image_dim, channels_data, along_radius, theta, final_output_path)

    # ------------------------------------------------------------------
    # STEP 6 — Shape analysis
    # ------------------------------------------------------------------
    sector_data  = None
    deform_score = None

    try:
        sector_data = analyze_vesicle_sectors(
            intensity_profiles_corrected,
            along_radius_full,
            theta,
            analysis_config
        )
        deform_score = calculate_deformability_score(
            intensity_profiles_corrected,
            along_radius_full,
            theta,
            pixel_size,
            analysis_config,
            sector_data=sector_data
        )

        if deform_score['quality_flag'] == 'questionable':
            comment.append(
                f"questionable_shape_sol{deform_score['solidity']:.4f}")

        # Uncomment to save detailed sector plots for every vesicle:
        # plot_sector_analysis(int(ves_coordinates[0]), sector_data,
        #                      deform_score, along_radius_full,
        #                      intensity_profiles_corrected, theta,
        #                      final_output_path, pixel_size)

    except Exception as e:
        print(f"  Warning: shape analysis failed for vesicle "
              f"{int(ves_coordinates[0])}: {e}")
        sector_data  = None
        deform_score = None

    # ------------------------------------------------------------------
    # GATE 1 + 2 — Exclude wide-peak artefacts and debris
    # ------------------------------------------------------------------
    if "wide_peak" in comment_peak or (refined_radius_um < min_radius_um):
        background       = np.zeros((1, num_channels))
        localization_val = np.zeros(num_channels - 1)
        lumen_val        = np.zeros(num_channels - 1)

        reason = "too_small" if refined_radius_um < min_radius_um else "wide_peak"
        if reason not in comment:
            comment.append(reason)

        comment_final = [', '.join(comment)] if comment else ["OK"]
        row = format_result_row(
            exp_info, ves_coordinates, background, localization_val,
            lumen_val, None, None, None, None, refined_radius_um, comment_final,
            shape_deform_dict=deform_score, shape_sector_data=sector_data)
        plt.close('all')
        return row, refined_radius_um

    # ------------------------------------------------------------------
    # GATE 3 — No membrane peak found
    # ------------------------------------------------------------------
    if death_mark_peak:
        background       = np.zeros((1, num_channels))
        localization_val = np.zeros(num_channels - 1)
        # lumen_val already initialised at the top of this function
        comment = comment_peak
        comment_final = [', '.join(comment)] if comment else ["OK"]
        row = format_result_row(
            exp_info, ves_coordinates, background, localization_val,
            lumen_val, None, None, None, None, refined_radius_um, comment_final,
            shape_deform_dict=deform_score, shape_sector_data=sector_data)
        plt.close('all')
        return row, refined_radius_um

    # ------------------------------------------------------------------
    # STEP 7 — Low-signal check
    # ------------------------------------------------------------------
    for j in range(num_channels - 1):
        if max(radial_profiles[:index_border_out, j + 1]) <= 5:
            comment.append("low_protein")

    # ------------------------------------------------------------------
    # STEP 8 — Angular (membrane-ring) profile
    # ------------------------------------------------------------------
    angular_profiles = angular_profile(
        num_channels, intensity_profiles_corrected,
        index_border_in, index_border_out, pixels_to_remove)

    quality_comments = check_membrane_quality(
        angular_profiles, radial_profiles, ves_coordinates[3])
    if quality_comments:
        comment += quality_comments

    if index_border_in == index_border_out:
        comment.append("no_memb_detected")
        comment_final = [', '.join(comment)] if comment else ["OK"]
        row = format_result_row(
            exp_info, ves_coordinates, background,
            np.zeros(num_channels - 1), lumen_val,
            None, None, None, None, refined_radius_um, comment_final,
            shape_deform_dict=deform_score, shape_sector_data=sector_data)
        plt.close('all')
        return row, refined_radius_um

    # ------------------------------------------------------------------
    # STEP 9 — Protein localization
    # ------------------------------------------------------------------
    size_central_area = parameters_sizes[2]
    localization_val, lumen_val, comment_loc = localization(
        num_channels, angular_profiles, radial_profiles,
        index_border_in, index_border_out,
        ves_coordinates[3], size_central_area, pixels_to_remove)

    # ------------------------------------------------------------------
    # STEP 10 — Actin structural metrics
    # ------------------------------------------------------------------
    actin_idx       = num_channels - 1
    loc_actin_score = localization_val[-1] if num_channels > 1 else 0.0

    t_cortex, ism, gini, rad_kurtosis = analyze_actin_structure(
        radial_profiles, angular_profiles, loc_actin_score,
        actin_idx, pixel_size, index_border_in, index_border_out)

    # ------------------------------------------------------------------
    # FINAL COMMENTS + OUTPUT
    # ------------------------------------------------------------------
    if np.isnan(localization_val).any():
        comment.append("no_memb_detected")
    elif np.isinf(localization_val).any():
        comment.append("zero_at_centre")
    if comment_loc:
        comment += comment_loc

    plot_intensity_profiles(
        plot_int_profiles, proteins_present, parameters_sizes,
        channels_data, ves_coordinates, image_dim,
        along_radius, theta, radial_profiles, angular_profiles,
        index_border_in, index_border_out,
        background, localization_val,
        final_output_path, exp_info, pixel_size)

    plt.close('all')

    comment_final = [', '.join(comment)] if comment else ["OK"]
    row = format_result_row(
        exp_info, ves_coordinates, background,
        localization_val, lumen_val,
        t_cortex, ism, gini, rad_kurtosis,
        refined_radius_um, comment_final,
        shape_deform_dict=deform_score,
        shape_sector_data=sector_data)

    return row, refined_radius_um