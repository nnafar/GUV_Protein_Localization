# -*- coding: utf-8 -*-
"""
guve_shape_analysis.py

Sector-Based Shape Analysis + Deformability Score for GUVs.

This module detects and quantifies deviations from spherical shape,
combining spatial sector information with a single summary score.

CHANGES FROM PREVIOUS VERSION:
-------------------------------
1. `along_radius` is now ACTUALLY USED in analyze_vesicle_sectors.
   Previously it was listed as a parameter but silently ignored.
   Now sector_radius = along_radius[argmax], which gives a real
   physical distance (in whatever unit along_radius is expressed,
   typically pixels from the centre). This means:
   - If dr=1, the value is in pixels (same as before, but correct)
   - If dr=2, the value is 2x the old index (previously WRONG)

2. `initial_radius` removed from calculate_deformability_score.
   It was listed as a parameter but never used anywhere in the body.
   Removing it makes the function signature honest.

3. Hard-coded thresholds (0.3, 3, 10.0, 0.15, 0.20 ...) are now
   read from an `analysis_config` dictionary passed in from main.py.
   You change those numbers in ONE place (main.py), not scattered
   through the code.

4. Peak detection in analyze_vesicle_sectors now uses BOTH a relative
   height threshold (fraction of max) AND an absolute minimum signal
   level. This prevents random noise in dim sectors from being counted
   as real membrane peaks.
"""

import numpy as np
from scipy.signal import find_peaks
import warnings


# ============================================================================
# SECTOR-BASED ANALYSIS
# ============================================================================

def analyze_vesicle_sectors(intensity_profiles, along_radius, theta,
                             analysis_config):
    """
    Divides the vesicle into pie-shaped sectors and analyzes each one.

    WHY THIS WORKS:
    ---------------
    Imagine cutting a pizza into 8 slices. We analyze each slice
    separately to detect if one part of the vesicle is different
    from the others. This helps us find:
      - Vesicles stuck together  → one sector has two radial peaks
      - Budding vesicles         → one sector bulges outward
      - Uneven protein coating   → one sector is much brighter

    HOW `along_radius` IS USED:
    ----------------------------
    The intensity array has shape (radial_steps, angles, channels).
    Each row corresponds to a specific distance from the centre.
    `along_radius` contains WHAT DISTANCE each row represents.

    Example: if dr=1,  along_radius = [0, 1, 2, 3, ...]   pixels
             if dr=2,  along_radius = [0, 2, 4, 6, ...]   pixels

    We find which row has the brightest signal (argmax), then look
    up its actual distance in along_radius.
    This is always correct regardless of your dr setting.

    PARAMETERS:
    -----------
    intensity_profiles : numpy array  shape (radial_steps, angles, channels)
        The 2D intensity profiles — radial distance changes along axis 0,
        angle changes along axis 1, channel along axis 2.

    along_radius : numpy array  shape (radial_steps,)
        The physical distance corresponding to each row in
        intensity_profiles. Units: same as dr in main.py (pixels).
        *** Must match the first dimension of intensity_profiles ***

    theta : numpy array  shape (angles,)
        The angle values (in radians) for each column.

    analysis_config : dict
        Configuration dictionary from main.py. Used keys:
          'num_sectors'                  - how many pie slices (default 8)
          'sector_peak_height_fraction'  - minimum peak height as fraction of max
          'sector_peak_min_signal'       - minimum absolute signal to count a peak
          'sector_peak_min_distance_px'  - minimum gap (pixels) between two peaks

    RETURNS:
    --------
    sector_data : list of dicts, one dict per sector.
        Each dict contains:
          'sector_idx'        : int   - sector number (0 to num_sectors-1)
          'sector_angle_range': tuple - (start_deg, end_deg)
          'radius'            : float - membrane distance in THIS sector
                                        in the same units as along_radius
                                        (previously was a raw array index)
          'radius_idx'        : int   - raw array index of the membrane peak
                                        (kept for internal use)
          'peak_count'        : int   - how many membrane peaks found
                                        (1 is ideal, >1 is suspicious)
          'intensity'         : float - maximum membrane brightness
          'symmetry_score'    : float - std of mean intensity across angles
                                        in this sector; lower = more uniform
          'has_multiple_peaks': bool  - True if more than one peak detected
          'mean_intensity'    : float - average brightness in this sector
    """

    # --- Pull settings from the config dictionary ---
    # The .get(key, default) pattern means: "use the value from config
    # if it exists, otherwise fall back to the default value shown".
    # This prevents crashes if someone passes an incomplete config.
    num_sectors       = analysis_config.get('num_sectors', 8)
    height_fraction   = analysis_config.get('sector_peak_height_fraction', 0.3)
    min_signal        = analysis_config.get('sector_peak_min_signal', 10.0)
    min_distance_px   = analysis_config.get('sector_peak_min_distance_px', 3)

    # How many degrees wide is each sector?
    # Example: 8 sectors → 360/8 = 45 degrees each
    angles_per_sector = 360.0 / num_sectors
    sector_data = []

    for sector_idx in range(num_sectors):

        # --- Step 1: Work out which angles belong to this sector ---

        # Degree boundaries (just for reporting)
        angle_start_deg = sector_idx * angles_per_sector
        angle_end_deg   = (sector_idx + 1) * angles_per_sector

        # Array index boundaries
        # We map from [0, num_sectors) to [0, len(theta))
        idx_start = int(sector_idx       * len(theta) / num_sectors)
        idx_end   = int((sector_idx + 1) * len(theta) / num_sectors)

        # --- Step 2: Extract the membrane channel for this sector ---
        # Result shape: (radial_steps, angles_in_this_sector)
        sector_membrane = intensity_profiles[:, idx_start:idx_end, 0]

        # --- Step 3: Collapse across angles to get one radial profile ---
        # Mean across all angles in the sector → shape: (radial_steps,)
        sector_profile = np.mean(sector_membrane, axis=1)

        # --- Step 4: Find the membrane position (radius) ---
        #
        # BEFORE (wrong when dr != 1):
        #   sector_radius = np.argmax(sector_profile)
        #   This gives an ARRAY INDEX (0, 1, 2, ...), not a distance.
        #   If dr=1 it happened to equal the distance in pixels,
        #   but if dr=2 the index would be half the real distance.
        #
        # AFTER (always correct):
        #   sector_radius_idx  = which row is brightest (index)
        #   sector_radius      = along_radius[that index] (real distance)
        #
        # Analogy: like looking up a word's page number in an index,
        # then turning to that page to read the actual content.
        sector_radius_idx = int(np.argmax(sector_profile))
        sector_radius     = along_radius[sector_radius_idx]

        # --- Step 5: Measure brightness uniformity across angles ---
        # For each angle in the sector, average across the radial dimension,
        # then measure how much those per-angle values vary.
        # Low std = uniform membrane, high std = one side much brighter
        sector_symmetry = np.std(np.mean(sector_membrane, axis=0))

        # --- Step 6: Maximum brightness in this sector ---
        sector_intensity = np.max(sector_membrane)

        # --- Step 7: Count membrane peaks in this sector ---
        #
        # WHY TWO THRESHOLDS?
        # The relative threshold (height_fraction) alone fails when the
        # sector is very dim (e.g. no membrane signal). If max = 2.0 a.u.
        # (pure noise), then 30% of max = 0.6, and random noise spikes
        # above 0.6 get counted as "membrane peaks". Adding an absolute
        # minimum (min_signal, e.g. 10.0) stops this.
        #
        # A peak must clear BOTH hurdles to be counted:
        #   1. At least height_fraction × max (relative)
        #   2. At least min_signal (absolute)
        # The max() call picks whichever threshold is higher.
        relative_threshold = np.max(sector_profile) * height_fraction
        effective_threshold = max(relative_threshold, min_signal)

        peaks, _ = find_peaks(
            sector_profile,
            height=effective_threshold,
            distance=min_distance_px
        )

        # --- Step 8: Store all results for this sector ---
        sector_data.append({
            'sector_idx'         : sector_idx,
            'sector_angle_range' : (angle_start_deg, angle_end_deg),
            'radius'             : sector_radius,       # ← physical distance
            'radius_idx'         : sector_radius_idx,   # ← raw array index
            'peak_count'         : len(peaks),
            'peaks_indices'      : peaks,
            'intensity'          : sector_intensity,
            'symmetry_score'     : sector_symmetry,
            'has_multiple_peaks' : len(peaks) > 1,
            'mean_intensity'     : np.mean(sector_membrane),
        })

    return sector_data


# ============================================================================
# DEFORMABILITY SCORE
# ============================================================================

def calculate_deformability_score(intensity_profiles, along_radius, theta,
                                  pixel_size, analysis_config,
                                  sector_data=None):
    """
    Creates a "deformability score" (0–1) summarising how much the vesicle
    deviates from a perfect sphere.

    SCORE INTERPRETATION:
    ---------------------
    0.00 – 0.15 : Good       — nearly perfect sphere
    0.15 – 0.30 : Acceptable — slightly deformed
    0.30 – 0.50 : Fair       — notable deformation
    0.50+       : Poor       — highly deformed, consider excluding

    CHANGES FROM PREVIOUS VERSION:
    -------------------------------
    - `initial_radius` parameter REMOVED (it was never used).
    - Score thresholds (0.15, 0.20) now come from `analysis_config`
      instead of being buried as magic numbers in the function body.
    - `along_radius` is now actually passed to analyze_vesicle_sectors
      so that per-sector radii are in physical units.

    PARAMETERS:
    -----------
    intensity_profiles : numpy array
        Full background-corrected 2D intensity data.
        Shape: (radial_steps, angles, channels).
        *** Must NOT be the sliced version — pass the full array ***

    along_radius : numpy array
        Radial distances, one value per row of intensity_profiles.
        *** Length must match intensity_profiles.shape[0] ***

    theta : numpy array
        Angle values (radians).

    pixel_size : float
        Micrometres per pixel (used for converting radius to µm in output).

    analysis_config : dict
        Configuration dictionary from main.py. See analyze_vesicle_sectors
        for which keys are used.

    sector_data : list of dicts, optional
        Pre-computed output from analyze_vesicle_sectors().
        If you pass this in, the function skips recomputing it.
        If None, it is computed here.

    RETURNS:
    --------
    dict with:
        solidity              : float 0–1, polygon_area / convex_hull_area
                                (1.0 = perfectly convex; lower = concavities)
        quality_flag          : str   'good' / 'acceptable' / 'questionable'
        radii_per_sector      : array of per-sector radius values (pixels)
        radii_per_sector_um   : array of per-sector radius values (µm)
        intensities_per_sector: array of per-sector intensities (visualisation only)
        mean_radius_um        : float, mean membrane radius in µm
        num_sectors           : int
    """

    # Compute sector data if not provided
    if sector_data is None:
        sector_data = analyze_vesicle_sectors(
            intensity_profiles, along_radius, theta, analysis_config
        )

    radii_per_sector    = np.array([s['radius'] for s in sector_data])
    radii_per_sector_um = radii_per_sector * pixel_size
    mean_radius         = np.mean(radii_per_sector)
    mean_radius_um      = mean_radius * pixel_size
    num_sectors         = len(sector_data)

    # Intensity per sector is kept for the optional visualisation plot only —
    # it is not written to the CSV.
    intensities_per_sector = np.array([s['intensity'] for s in sector_data])

    # -----------------------------------------------------------------------
    # SOLIDITY  (replaces the old weighted Deformability_Score)
    # -----------------------------------------------------------------------
    # Solidity = polygon_area / convex_hull_area
    #
    # Imagine the 8 sector membrane positions as dots on a map.
    # Connect them in order → you get an 8-sided polygon (the actual
    # membrane outline).  Now stretch a rubber band around the outside
    # → you get the convex hull.  Solidity asks:
    #   "How much of the rubber-band area is actually filled by membrane?"
    #
    #   1.0  = perfectly convex (rubber band hugs the membrane exactly)
    #   <1.0 = concavities present: dents, budding vesicles, aggregation
    #
    # A single sector that points inward (because a second vesicle is
    # pressing in from outside) lowers solidity noticeably — exactly the
    # kind of artefact we want to flag.
    #
    # Pull thresholds from analysis_config so they can be tuned in main.py
    # without touching this file.
    solidity_good_thresh       = analysis_config.get('solidity_good_threshold',       0.95)
    solidity_acceptable_thresh = analysis_config.get('solidity_acceptable_threshold', 0.85)

    try:
        from scipy.spatial import ConvexHull

        # Build the sector angles (in radians) from the sector data.
        # Each sector_angle_range[0] is the START angle of that pie-slice.
        angles_rad = np.array(
            [s['sector_angle_range'][0] * np.pi / 180.0 for s in sector_data]
        )

        # Convert polar (r, θ) → Cartesian (x, y) for each sector
        xs = radii_per_sector * np.cos(angles_rad)
        ys = radii_per_sector * np.sin(angles_rad)

        # Shoelace formula: area of the (ordered) polygon formed by the
        # 8 membrane positions.
        # Works because sectors are already in angular order (0° → 315°).
        polygon_area = 0.5 * abs(
            np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)
        )

        # Convex hull area.  In scipy, ConvexHull.volume = area in 2-D.
        hull = ConvexHull(np.column_stack([xs, ys]))
        convex_area = hull.volume   # area for 2-D data

        solidity = (polygon_area / convex_area) if convex_area > 0 else 1.0

    except Exception:
        # Graceful fallback: if the convex hull fails (e.g. all 8 sector
        # radii are identical → degenerate polygon), assume perfect solidity
        # so the vesicle is not incorrectly flagged.
        solidity = 1.0

    # Safety clamp — floating-point rounding can push values very slightly
    # outside [0, 1]; clip to keep the value interpretable.
    solidity = float(np.clip(solidity, 0.0, 1.0))

    # Assign a human-readable quality flag.
    # Note the direction is INVERTED relative to the old total_score:
    #   total_score: higher = worse (deformation score)
    #   solidity:    higher = better (1.0 = perfect sphere)
    if solidity >= solidity_good_thresh:
        quality_flag = 'good'
    elif solidity >= solidity_acceptable_thresh:
        quality_flag = 'acceptable'
    else:
        quality_flag = 'questionable'

    return {
        'solidity'             : solidity,
        'quality_flag'         : quality_flag,
        'radii_per_sector'     : radii_per_sector,
        'radii_per_sector_um'  : radii_per_sector_um,
        'intensities_per_sector': intensities_per_sector,   # for visualisation
        'mean_radius_um'       : mean_radius_um,
        'num_sectors'          : num_sectors,
    }


# ============================================================================
# VISUALISATION
# ============================================================================

def plot_sector_analysis(ves_id, sector_data, deform_score, along_radius,
                         intensity_profiles, theta, final_output_path,
                         pixel_size):
    """
    Creates a detailed visualisation of the sector analysis.

    Shows:
      1. Polar bar chart  — membrane radius per sector
      2. Bar chart        — membrane intensity per sector
      3. Summary text     — all key statistics
      4. Radial profile   — averaged across all angles

    PARAMETERS:
    -----------
    ves_id           : int    — vesicle ID for the filename
    sector_data      : list   — output of analyze_vesicle_sectors()
    deform_score     : dict   — output of calculate_deformability_score()
    along_radius     : array  — radial distances (pixels)
    intensity_profiles: array — full 2D intensity data
    theta            : array  — angles in radians
    final_output_path: str    — folder to save the plot
    pixel_size       : float  — µm per pixel
    """
    import matplotlib.pyplot as plt
    import os

    fig = plt.figure(figsize=(14, 10), dpi=125)
    gs  = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: Polar bar chart of radius per sector ---
    ax1     = fig.add_subplot(gs[0, 0], projection='polar')
    # Radii are now in pixels (along_radius units), convert to µm for display
    radii   = deform_score['radii_per_sector_um']
    angles  = np.array([s['sector_angle_range'][0] for s in sector_data])
    angles_rad = angles * np.pi / 180.0
    colors  = ['red' if s['has_multiple_peaks'] else 'steelblue'
               for s in sector_data]

    ax1.bar(angles_rad, radii, width=0.7, color=colors, alpha=0.7,
            edgecolor='black')
    ax1.set_title('Membrane Radius per Sector (µm)\n(Red = multiple peaks)',
                  pad=20)
    if len(radii) > 0 and max(radii) > 0:
        ax1.set_ylim(0, max(radii) * 1.1)

    # --- Panel 2: Intensity per sector ---
    ax2         = fig.add_subplot(gs[0, 1])
    sector_nums = np.arange(len(sector_data))
    intensities = deform_score['intensities_per_sector']

    ax2.bar(sector_nums, intensities, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(np.mean(intensities), color='green', linestyle='--',
                linewidth=2, label='Mean')
    ax2.set_xlabel('Sector Number')
    ax2.set_ylabel('Membrane Intensity (a.u.)')
    ax2.set_title('Membrane Intensity per Sector')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # --- Panel 3: Summary statistics text ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    summary_text = (
        f"VESICLE {ves_id}: SHAPE ANALYSIS SUMMARY\n\n"
        f"Solidity: {deform_score['solidity']:.4f}"
        f"  ({deform_score['quality_flag'].upper()})\n"
        f"  (sectors with mult. peaks: "
        f"{sum(1 for s in sector_data if s['has_multiple_peaks'])})\n\n"
        f"Radius per Sector (µm):\n"
        f"  Mean:    {np.mean(radii):.2f} µm\n"
        f"  Std Dev: {np.std(radii):.2f} µm\n"
        f"  Range:   {np.min(radii):.2f} – {np.max(radii):.2f} µm\n\n"
        f"INTERPRETATION:\n"
        f"  Score < 0.15         → Good (nearly spherical)\n"
        f"  Score 0.15 – 0.30    → Acceptable (slightly deformed)\n"
        f"  Score > 0.30         → Questionable (consider review)"
    )

    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # --- Panel 4: Radial profile averaged across all angles ---
    ax4              = fig.add_subplot(gs[2, :])
    membrane_profile = np.mean(intensity_profiles[:, :, 0], axis=1)
    radius_um_axis   = along_radius * pixel_size

    ax4.plot(radius_um_axis, membrane_profile, linewidth=2, color='steelblue',
             label='Membrane Profile (mean across all angles)')
    ax4.set_xlabel('Radius (µm)')
    ax4.set_ylabel('Intensity (a.u.)')
    ax4.set_title('Radial Profile (Averaged Across All Angles)')
    ax4.grid(alpha=0.3)
    ax4.legend()

    os.makedirs(final_output_path, exist_ok=True)
    filename = f"Vesicle_{ves_id}_Shape_Analysis.png"
    fig.savefig(os.path.join(final_output_path, filename),
                dpi=125, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# CSV HELPER
# ============================================================================

def format_shape_metrics_for_csv(deform_score, sector_data):
    """
    Converts deformability results into a flat list for CSV output.

    Returns values in this order (matching the column headers in skeleton.py):
      [Solidity, Shape_Quality_Flag, Sector_Details]

    The Sector_Details field now reports the mean radius in µm
    (instead of pixels) so the value is interpretable without
    needing to look up pixel_size separately.

    PARAMETERS:
    -----------
    deform_score : dict  — output of calculate_deformability_score()
    sector_data  : list  — output of analyze_vesicle_sectors()

    RETURNS:
    --------
    list of 3 values ready to write as CSV columns
    """
    mean_radius_um = float(np.mean(deform_score['radii_per_sector_um']))

    csv_values = [
        round(float(deform_score['solidity']), 4),
        deform_score['quality_flag'],
        f"sectors(r_mean_um={mean_radius_um:.2f},n_sectors={deform_score['num_sectors']})",
    ]

    return csv_values