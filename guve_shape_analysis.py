# -*- coding: utf-8 -*-
"""
guve_shape_analysis.py

NEW MODULE: Sector-Based Analysis + Deformability Score
This module detects and quantifies deviations from spherical shape
in GUVs, combining spatial sector information with a summary score.

Author: Enhanced by Claude
Date: 2025
"""

import numpy as np
from scipy.signal import find_peaks
import warnings

# ============================================================================
# SOLUTION 3 + 4: SECTOR-BASED ANALYSIS + DEFORMABILITY SCORE
# ============================================================================

def analyze_vesicle_sectors(intensity_profiles, along_radius, theta, num_sectors=8):
    """
    Divides the vesicle into pie-shaped sectors and analyzes each one.
    
    WHY THIS WORKS:
    Imagine cutting a pizza into 8 slices. We analyze each slice separately
    to detect if one part of the vesicle is different from the others.
    This helps us find:
    - Vesicles stuck together (one sector has two peaks)
    - Budding vesicles (one sector bulges out)
    - Uneven protein distribution (protein only on one side)
    
    PARAMETERS:
    -----------
    intensity_profiles : numpy array of shape (radial_distance, angle, channel)
        The 2D intensity profiles at each angle
        
    along_radius : numpy array of shape (radial_distance,)
        The radial distances corresponding to intensity_profiles
        
    theta : numpy array of shape (angle,)
        The angles (in radians) for each direction around the vesicle
        
    num_sectors : int
        How many pie slices to divide the vesicle into (default 8)
        More sectors = more detail but noisier, 8 is a good balance
    
    RETURNS:
    --------
    sector_data : list of dictionaries
        One dictionary per sector with:
        - sector_angle_range: (start_angle_deg, end_angle_deg)
        - radius: pixel distance to membrane in this sector
        - peak_count: how many membrane peaks found (1 is good, >1 is suspect)
        - intensity: maximum membrane brightness in this sector
        - symmetry_score: lower = more uniform across the sector
        - has_multiple_peaks: True if suspicious (might be artifact)
    
    EXAMPLE OUTPUT:
    ---------------
    sector_data[0] = {
        'sector_angle_range': (0.0, 45.0),     # 0-45 degrees
        'radius': 45.2,                        # Membrane is 45.2 pixels away
        'peak_count': 1,                       # One membrane peak (good)
        'intensity': 125.3,                    # Brightness level
        'symmetry_score': 3.2,                 # Variation within sector
        'has_multiple_peaks': False
    }
    """
    
    # Calculate how many degrees per sector
    # If 8 sectors, each is 360/8 = 45 degrees
    angles_per_sector = 360 / num_sectors
    sector_data = []
    
    # Loop through each sector
    for sector_idx in range(num_sectors):
        # Calculate the angular boundaries for this sector (in degrees)
        angle_start = sector_idx * angles_per_sector
        angle_end = (sector_idx + 1) * angles_per_sector
        
        # Convert degree ranges to array indices
        # (since theta is in radians, and array indices go from 0 to len(theta))
        idx_start = int(sector_idx * len(theta) / num_sectors)
        idx_end = int((sector_idx + 1) * len(theta) / num_sectors)
        
        # Extract just the membrane channel (channel 0) for this angular sector
        # Shape: (radial_distance, angles_in_this_sector)
        sector_membrane = intensity_profiles[:, idx_start:idx_end, 0]
        
        # --- Analyze this sector ---
        
        # 1. Find the typical radius in this sector
        #    Average across all angles in this sector
        sector_profile = np.mean(sector_membrane, axis=1)  # Average across angles
        sector_radius = np.argmax(sector_profile)  # Where is it brightest?
        
        # 2. Measure how consistent the membrane is within this sector
        #    If all angles in the sector have similar intensity, symmetry_score is LOW
        #    If some angles are very bright and others dark, symmetry_score is HIGH
        sector_symmetry = np.std(np.mean(sector_membrane, axis=0))
        
        # 3. What's the maximum brightness in this sector?
        sector_intensity = np.max(sector_membrane)
        
        # 4. Are there multiple membrane peaks in this sector?
        #    One peak = normal GUV
        #    Multiple peaks = might be two vesicles stuck together
        #    This is a warning sign that needs investigation
        
        # Find all peaks in the radial profile of this sector
        # "height" parameter: only count peaks that are at least 30% of max
        peaks, _ = find_peaks(
            sector_profile, 
            height=np.max(sector_profile) * 0.3,
            distance=3  # Peaks must be at least 3 pixels apart
        )
        
        # Store all the information for this sector
        sector_data.append({
            'sector_idx': sector_idx,
            'sector_angle_range': (angle_start, angle_end),
            'radius': sector_radius,
            'peak_count': len(peaks),
            'peaks_indices': peaks,  # Store actual peak positions
            'intensity': sector_intensity,
            'symmetry_score': sector_symmetry,
            'has_multiple_peaks': len(peaks) > 1,  # True if suspicious
            'mean_intensity': np.mean(sector_membrane)
        })
    
    return sector_data


def calculate_deformability_score(intensity_profiles, along_radius, theta, 
                                  initial_radius, pixel_size, sector_data=None):
    """
    Creates a comprehensive "deformability score" (0-1) that summarizes
    how much the vesicle deviates from a perfect sphere.
    
    SCORING SYSTEM:
    ---------------
    0.0-0.15 : Excellent - nearly perfect sphere
    0.15-0.30 : Good - slightly deformed, acceptable
    0.30-0.50 : Fair - notable deformation, might affect results
    0.50+    : Poor - highly deformed, consider excluding
    
    PARAMETERS:
    -----------
    intensity_profiles : numpy array
        2D intensity profile data
        
    along_radius : numpy array
        Radial distance values
        
    theta : numpy array
        Angular values (in radians)
        
    initial_radius : float
        The detected radius of the vesicle
        
    pixel_size : float
        Size of each pixel in micrometers
        
    sector_data : list of dicts, optional
        Output from analyze_vesicle_sectors()
        If None, we'll calculate it here
    
    RETURNS:
    --------
    deform_dict : dictionary with:
        - total_score: single number 0-1 summarizing overall deformation
        - radial_bumpiness: how much radius varies (0-1)
        - sector_uniformity: how consistent sectors are (0-1)
        - clustering_risk: likelihood of stuck vesicles (0-1)
        - quality_flag: "good", "acceptable", or "questionable"
        - radii_per_sector: actual measured radius in each sector
        - intensities_per_sector: membrane brightness in each sector
    
    EXAMPLE OUTPUT:
    ---------------
    {
        'total_score': 0.18,                 # Overall shape quality
        'radial_bumpiness': 0.12,            # Radius varies by ~12%
        'sector_uniformity': 0.08,           # Sectors similar
        'clustering_risk': 0.05,             # Low risk of sticking
        'quality_flag': 'good',
        'radii_per_sector': [45, 46, 44, 47, 45, 44, 46, 45],
        'intensities_per_sector': [120, 122, 119, 123, ...]
    }
    """
    
    # Calculate sector data if not provided
    if sector_data is None:
        sector_data = analyze_vesicle_sectors(intensity_profiles, along_radius, theta, num_sectors=8)
    
    # --- COMPONENT 1: Radial Bumpiness ---
    # How much does the membrane distance vary around the circle?
    # Low variation = smooth circle, high variation = bumpy
    
    # Get the radius in each sector
    radii_per_sector = np.array([s['radius'] for s in sector_data])
    mean_radius = np.mean(radii_per_sector)
    std_radius = np.std(radii_per_sector)
    
    # Coefficient of Variation = std/mean
    # This is a standardized measure of how much things vary
    # 0.05 = 5% variation (very smooth)
    # 0.20 = 20% variation (bumpy)
    # 0.50+ = very bumpy
    if mean_radius > 0:
        cv_radius = std_radius / mean_radius
    else:
        cv_radius = 0
    
    # Scale to 0-1, where 0.15 maps to 1.0 (bumpiness threshold)
    radial_component = min(cv_radius / 0.15, 1.0)
    
    # --- COMPONENT 2: Sector Uniformity ---
    # Are all sectors equally bright?
    # If one side is much brighter than the other, it might be deformed or stuck
    
    intensities_per_sector = np.array([s['intensity'] for s in sector_data])
    mean_intensity = np.mean(intensities_per_sector)
    
    if mean_intensity > 0:
        # Coefficient of variation for intensity
        sector_cv = np.std(intensities_per_sector) / mean_intensity
    else:
        sector_cv = 0
    
    # Scale to 0-1, where 0.20 maps to 1.0 (uniformity threshold)
    sector_component = min(sector_cv / 0.20, 1.0)
    
    # --- COMPONENT 3: Clustering Risk ---
    # Do we see multiple peaks in any sector?
    # This suggests two vesicles stuck together or membrane artifacts
    
    multi_peak_sectors = sum(1 for s in sector_data if s['has_multiple_peaks'])
    num_sectors = len(sector_data)
    
    # If more than 1 sector has multiple peaks, it's a red flag
    clustering_component = min(multi_peak_sectors / num_sectors * 2, 1.0)
    
    # --- COMBINE ALL COMPONENTS ---
    # Each component has a weight showing how important it is
    # Total = weighted sum of all components
    
    total_score = (
        0.5 * radial_component +      # Bumpiness: 50% weight (most important)
        0.3 * sector_component +      # Sector uniformity: 30%
        0.2 * clustering_component    # Clustering: 20%
    )
    
    # Assign quality flag based on score
    if total_score < 0.15:
        quality_flag = 'good'
    elif total_score < 0.30:
        quality_flag = 'acceptable'
    else:
        quality_flag = 'questionable'
    
    # Return everything as a dictionary
    return {
        'total_score': total_score,
        'radial_bumpiness': radial_component,
        'sector_uniformity': sector_component,
        'clustering_risk': clustering_component,
        'quality_flag': quality_flag,
        'radii_per_sector': radii_per_sector,
        'intensities_per_sector': intensities_per_sector,
        'num_sectors': num_sectors,
        'coefficient_of_variation_radius': cv_radius,
        'coefficient_of_variation_intensity': sector_cv
    }


def plot_sector_analysis(ves_id, sector_data, deform_score, along_radius, 
                        intensity_profiles, theta, final_output_path, pixel_size):
    """
    Creates a detailed visualization of the sector analysis.
    
    Shows:
    1. A pie chart of radius variations
    2. A bar chart of intensity per sector
    3. Highlights problematic sectors
    
    PARAMETERS:
    -----------
    ves_id : int
        Vesicle ID for naming the plot
        
    sector_data : list of dicts
        Output from analyze_vesicle_sectors()
        
    deform_score : dict
        Output from calculate_deformability_score()
        
    along_radius : numpy array
        Radial distances
        
    intensity_profiles : numpy array
        Full 2D intensity data
        
    theta : numpy array
        Angles (radians)
        
    final_output_path : str
        Where to save the plot
        
    pixel_size : float
        Pixel size in micrometers
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig = plt.figure(figsize=(14, 10), dpi=125)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # --- Panel 1: Pie chart of radius per sector ---
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    
    radii = deform_score['radii_per_sector']
    angles = np.array([s['sector_angle_range'][0] for s in sector_data])
    angles_rad = angles * np.pi / 180
    
    # Color problematic sectors differently
    colors = ['red' if s['has_multiple_peaks'] else 'steelblue' 
              for s in sector_data]
    
    ax1.bar(angles_rad, radii, width=0.7, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Radius per Sector\n(Red = multiple peaks detected)', pad=20)
    ax1.set_ylim(0, max(radii) * 1.1)
    
    # --- Panel 2: Bar chart of intensity per sector ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    sector_nums = np.arange(len(sector_data))
    intensities = deform_score['intensities_per_sector']
    colors = ['red' if s['has_multiple_peaks'] else 'steelblue' 
              for s in sector_data]
    
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
    
    summary_text = f"""
VESICLE {ves_id}: SHAPE ANALYSIS SUMMARY

Overall Deformability Score: {deform_score['total_score']:.3f} ({deform_score['quality_flag'].upper()})
    • Radial Bumpiness:        {deform_score['radial_bumpiness']:.3f}  (radius variation: {deform_score['coefficient_of_variation_radius']*100:.1f}%)
    • Sector Uniformity:       {deform_score['sector_uniformity']:.3f}  (intensity variation: {deform_score['coefficient_of_variation_intensity']*100:.1f}%)
    • Clustering Risk:         {deform_score['clustering_risk']:.3f}  (sectors with multiple peaks: {sum(1 for s in sector_data if s['has_multiple_peaks'])})

Radius Statistics:
    • Mean:    {np.mean(radii):.1f} pixels  ({np.mean(radii)*pixel_size:.2f} µm)
    • Std Dev: {np.std(radii):.1f} pixels   ({np.std(radii)*pixel_size:.2f} µm)
    • Range:   {np.min(radii):.1f} - {np.max(radii):.1f} pixels
    
Intensity Statistics:
    • Mean:    {np.mean(intensities):.1f} a.u.
    • Std Dev: {np.std(intensities):.1f} a.u.
    • Range:   {np.min(intensities):.1f} - {np.max(intensities):.1f} a.u.

INTERPRETATION:
    • Deformability < 0.15:  Good - nearly spherical ✓
    • Deformability 0.15-0.30: Acceptable - slightly deformed
    • Deformability > 0.30:  Questionable - highly deformed (⚠️ consider review)
    """
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # --- Panel 4: Radial profile with sector boundaries ---
    ax4 = fig.add_subplot(gs[2, :])
    
    membrane_profile = np.mean(intensity_profiles[:, :, 0], axis=1)
    radius_um = along_radius * pixel_size
    
    ax4.plot(radius_um, membrane_profile, linewidth=2, color='steelblue', label='Membrane Profile')
    
    # Shade regions corresponding to different sectors
    sector_angles = np.array([s['sector_angle_range'][0] for s in sector_data])
    colors_sector = plt.cm.Set3(np.linspace(0, 1, len(sector_data)))
    
    ax4.set_xlabel('Radius (µm)')
    ax4.set_ylabel('Intensity (a.u.)')
    ax4.set_title('Radial Profile (Averaged Across All Angles)')
    ax4.grid(alpha=0.3)
    ax4.legend()
    
    # Save the figure
    import os
    os.makedirs(final_output_path, exist_ok=True)
    filename = f"Vesicle_{ves_id}_Shape_Analysis.png"
    fig.savefig(os.path.join(final_output_path, filename), dpi=125, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# HELPER FUNCTION: Add shape metrics to your CSV row
# ============================================================================

def format_shape_metrics_for_csv(deform_score, sector_data):
    """
    Converts deformability results into values suitable for CSV output.
    
    Returns a list of values in the order expected by your CSV headers:
    [deformability_score, radial_bumpiness, sector_uniformity, 
     clustering_risk, quality_flag, sector_details]
    
    PARAMETERS:
    -----------
    deform_score : dict
        Output from calculate_deformability_score()
        
    sector_data : list of dicts
        Output from analyze_vesicle_sectors()
    
    RETURNS:
    --------
    csv_values : list
        Values ready to write to CSV
    """
    
    # Round to reasonable precision for CSV
    csv_values = [
        round(float(deform_score['total_score']), 4),
        round(float(deform_score['radial_bumpiness']), 4),
        round(float(deform_score['sector_uniformity']), 4),
        round(float(deform_score['clustering_risk']), 4),
        deform_score['quality_flag'],
        # Optional: sector details as a compact string
        f"sectors(r_mean={np.mean(deform_score['radii_per_sector']):.1f},n_peaks={sum(1 for s in sector_data if s['has_multiple_peaks'])})"
    ]
    
    return csv_values

