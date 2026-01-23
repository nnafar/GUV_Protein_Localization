# -*- coding: utf-8 -*-
"""
skeleton.py

Contains all necessary functions for the GUV analysis pipeline.
Updated to include:
- Automatic pixel size detection (tifffile)
- Robust localization calculation (np.max)
- Actin cortical thickness and density metrics
- Physical unit plotting (microns)
- FIX: 'Lumenal Actin' logic (Preserves low density, checks peak position for thickness)

@author: kkourkoulou (Updated by Gemini)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap 
import pandas as pd                                                             
import csv               
          
from skimage.filters import try_all_threshold                                  
from skimage.filters import threshold_li, threshold_otsu, threshold_yen, \
                            threshold_isodata, threshold_mean, threshold_minimum, threshold_triangle
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

from scipy import ndimage as nd                                                
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import map_coordinates, gaussian_filter1d               
import cv2
import tifffile  # Required for metadata reading


def find_projects_info(path_membrane):
    """
    Finds the experiment information of all the data sets to be analyzed
    by scanning the specific Membrane (C1) directory.
    """
    set_of_data  = 0
    exp_info_all_sets = np.zeros((4), dtype = str)
    
    for f in os.listdir(path_membrane):
        
        if f.endswith("C1.tif"):
            
            name, extension = os.path.splitext(f)
            # Split filename: "20251110_BranchedCortex-Region0000-C1"
            parts = name.split("-")
            
            # Remove "C1" -> ["20251110_BranchedCortex", "Region0000"]
            parts.pop(-1)
            
            # Recombine -> "20251110_BranchedCortex-Region0000"
            full_name = "-".join(parts)
            
            # Construct info array: [Date/Name, Region, FullName, FullName]
            exp_info_single_set = [parts[0], parts[1], full_name, full_name]
            
            if set_of_data == 0:
                exp_info_all_sets = np.array(exp_info_single_set)
            else:
                exp_info_all_sets = np.vstack((exp_info_all_sets, exp_info_single_set))
            
            set_of_data += 1
    
    return exp_info_all_sets                     

def create_directory_structure(path_to_output_root, exp_info):
    """
    Creates a hierarchical folder structure: Root -> Date_Experiment -> Region_ID
    """
    date_experiment = exp_info[0] # e.g., 20251110_BranchedCortex
    region_id = exp_info[1]       # e.g., Region0000
    
    # Create the full path
    final_output_path = os.path.join(path_to_output_root, date_experiment, region_id)
    
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
        
    return final_output_path

def read_files(path_membrane, path_septin, path_actin, path_detected, exp_info, proteins_present):
    """
    Function that reads the image files and the vesicle detection from separate folders.
    UPDATED: Now detects pixel size from metadata.
    """
    file_stem = exp_info[3]
    
    # 1. Read Membrane (C1) - Always present
    c1_path = os.path.join(path_membrane, file_stem + "-C1.tif")
    channels_data = plt.imread(c1_path)
    
    # --- Detect Pixel Size ---
    pixel_size = 0.07 # Default fallback
    try:
        with tifffile.TiffFile(c1_path) as tif:
            # Check ImageJ metadata (common for Fiji exports)
            imagej_metadata = tif.imagej_metadata
            if imagej_metadata and 'spacing' in imagej_metadata:
                pixel_size = float(imagej_metadata['spacing'])
            # Check standard XResolution tags
            elif tif.pages[0].tags.get('XResolution'):
                x_res = tif.pages[0].tags['XResolution'].value
                # If unit is cm (3), convert to um. Res is px/unit. Size = 1/Res.
                if tif.pages[0].tags.get('ResolutionUnit').value == 3: #
                    pixel_size = 10000.0 / (x_res[0]/x_res[1])
                # Note: If unit is 'Inch' or 'None', logic may vary, keeping default
    except:
        print(f"Warning: Could not detect pixel size for {file_stem}. Using 1.0.")
    
    print(f"Processing: {file_stem} (Pixel Size: {pixel_size:.4f} um)")
    
    # 2. Read Septin (C2) - If present
    if proteins_present[0] == True:
        c2_path = os.path.join(path_septin, file_stem + "-C2.tif")
        if os.path.exists(c2_path):
            channels_data = np.dstack((channels_data, plt.imread(c2_path)))
        else:
            print(f"Warning: Septin file not found at {c2_path}")
        
    # 3. Read Actin (C3) - If present
    if proteins_present[1] == True: 
        c3_path = os.path.join(path_actin, file_stem + "-C3.tif")
        if os.path.exists(c3_path):
            channels_data = np.dstack((channels_data, plt.imread(c3_path)))
        else:
            print(f"Warning: Actin file not found at {c3_path}")
    
    # 4. Read Detected Vesicles CSV
    csv_path = os.path.join(path_detected, file_stem + "-detected_vesicles.csv")
    detected_vesicles = pd.read_csv(csv_path)
    
    # Create coordinates array [ID, xc, yc, radius]
    coordinates = np.zeros((len(detected_vesicles), 4))
    
    # ID, xc, yc, radius
    coordinates[:, 0] = np.arange(1, len(detected_vesicles) + 1)
    coordinates[:, 1] = detected_vesicles.iloc[:, 0].values      # x coordinate
    coordinates[:, 2] = detected_vesicles.iloc[:, 1].values      # y coordinate
    coordinates[:, 3] = detected_vesicles.iloc[:, 2].values / 2  # convert diameter to radius
    
    return channels_data, coordinates, pixel_size


def create_output_file(final_output_path, proteins_present):
    """
    Creates an empty output CSV file in the specific folder.
    UPDATED: Added headers for new metrics (t_cortex, rho_actin, uniformity, refined radius).
    """
    protein_status = plot_format(proteins_present)
    
    base_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius"]
    
    if protein_status == "both_proteins":
        specific_headers = ["M Background", "S Background", "A Background", 
                            "S localization", "A localization", 
                            "t_cortex", "rho_actin", "uniformity", 
                            "Refined Radius (um)", "Comment"] 

    if protein_status == "only_septin":
        specific_headers = ["M Background", "S Background", "S localization", 
                            "Refined Radius (um)", "Comment"] 

    if protein_status == "only_actin":
        specific_headers = ["M Background", "A Background", "A localization", 
                            "t_cortex", "rho_actin", "uniformity", 
                            "Refined Radius (um)", "Comment"] 

    column_headers = base_headers + specific_headers

    output_csv_path = os.path.join(final_output_path, "Analysis_Results.csv")
    
    with open(output_csv_path, "w", newline='') as output_file:
        df = csv.DictWriter(output_file, delimiter=',', fieldnames = column_headers)
        df.writeheader()
        
    return output_csv_path
        
    
def color_maps():
    """
    Creates and registers in matplotlib the colormaps.
    """
    if "cmap_cyan" not in plt.colormaps():
        colors = ["black", "cyan"]
        nodes = [0.0, 1.0]
        cmap_cyan = LinearSegmentedColormap.from_list("cmap_cyan", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_cyan) 
        
    if "cmap_magenta" not in plt.colormaps():
        colors = ["black","magenta"]
        nodes = [0.0, 1.0]
        cmap_magenta = LinearSegmentedColormap.from_list("cmap_magenta", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_magenta) 
    
    if "cmap_magenta_enhanced" not in plt.colormaps():
        colors = ["black","magenta", "magenta"]
        nodes = [0.0, 0.2, 1.0]
        cmap_magenta_enhanced = LinearSegmentedColormap.from_list("cmap_magenta_enhanced", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_magenta_enhanced) 
        
    if "cmap_yellow" not in plt.colormaps():
        colors = ["black","yellow"]
        nodes = [0.0, 1.0]
        cmap_yellow = LinearSegmentedColormap.from_list("cmap_yellow", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_yellow)

    
def detected_centres(to_plot, num_vesicles, ch_membrane, all_xc, all_yc, final_output_path, exp_info):
    """
    Displays the image to be analyzed with the centers of the vesicles.
    """
    if to_plot == True:
        plt.figure(dpi=125)
        plt.title('Total number of detected vesicles: ' + str(num_vesicles))    
        plt.axis('off')
        plt.imshow(ch_membrane, cmap="cmap_cyan")
        plt.scatter(all_xc, all_yc, color='red', s=1, marker="o")
        for i in range(num_vesicles):
            plt.annotate(i+1, (all_xc[i]+5, all_yc[i]-5), c = 'red') 
        
        plt.savefig(os.path.join(final_output_path, "Overview_Detected_Centres.png"))


def zoom_in_vesicle(channel, size_side, image_dim, ves_coordinates):
    """
    Creates a square cropped view of a chosen vesicle. 
    Correctly handles padding so the vesicle remains centered.
    """
    xc               = ves_coordinates[1]
    yc               = ves_coordinates[2]
    radius           = ves_coordinates[3]
    
    vesicle_box_side = int(size_side * radius)
    expected_size    = 2 * vesicle_box_side
    
    y_start_ideal    = int(yc - vesicle_box_side)
    y_end_ideal      = int(yc + vesicle_box_side)
    x_start_ideal    = int(xc - vesicle_box_side)
    x_end_ideal      = int(xc + vesicle_box_side)
    
    # Define the valid slice within the image dimensions
    y_start_valid    = max(0, y_start_ideal)
    y_end_valid      = min(image_dim[1], y_end_ideal)
    x_start_valid    = max(0, x_start_ideal)
    x_end_valid      = min(image_dim[0], x_end_ideal)
    
    # Perform the slice
    vesicle_slice = channel[y_start_valid:y_end_valid, x_start_valid:x_end_valid]
    
    # Check if we need padding
    if vesicle_slice.shape != (expected_size, expected_size):
        padded_box = np.zeros((expected_size, expected_size), dtype=channel.dtype)
        # Calculate where to paste the slice in the padded box
        # Offset is determined by how much we clipped from the left/top
        paste_y, paste_x = max(0, -y_start_ideal), max(0, -x_start_ideal)
        h_slice, w_slice = vesicle_slice.shape
        padded_box[paste_y:paste_y+h_slice, paste_x:paste_x+w_slice] = vesicle_slice
        
        return padded_box
    
    return vesicle_slice

def define_threshold(user_choice, channel, size_side, image_dim, ves_coordinates):    
    """
    Allows for optional manual choice of the threshold method.
    """
    vesicle_box_channel = zoom_in_vesicle(channel, size_side, image_dim, ves_coordinates)
    
    if user_choice == True:
        fig, ax = try_all_threshold(vesicle_box_channel, figsize=(5, 10), verbose=False)             
        
        threshold_method = input("Choose threshold method to follow:")
        
        methods = {'Isodata': threshold_isodata, 'Li': threshold_li, 'Mean': threshold_mean,
                   'Minimum': threshold_minimum, 'Otsu': threshold_otsu, 
                   'Triangle': threshold_triangle, 'Yen': threshold_yen}
        
        threshold = methods.get(threshold_method, threshold_li)(vesicle_box_channel)
    else:
        threshold = threshold_li(vesicle_box_channel)
        
    return threshold

def refine_guv_center(image, center_guess, radius_estimate, search_box_factor=1.5):
    """
    Refines the center using gradients and Hough transform to handle nearby vesicles.
    """
    xc, yc = center_guess
    box_half_width = int(radius_estimate * search_box_factor)
    
    # 1. ROI Extraction
    x_start, x_end = int(max(0, xc - box_half_width)), int(min(image.shape[1], xc + box_half_width))
    y_start, y_end = int(max(0, yc - box_half_width)), int(min(image.shape[0], yc + box_half_width))
    
    if x_start >= x_end or y_start >= y_end: return center_guess
    
    roi = image[y_start:y_end, x_start:x_end]
    roi_center_x = xc - x_start
    roi_center_y = yc - y_start
    
    if roi.size == 0: return center_guess

    # 2. Pre-processing
    roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    roi_eq = clahe.apply(roi_norm)
    roi_smooth = cv2.GaussianBlur(roi_eq, (5, 5), 0)

    # 3. Edges & Gradients
    edges = cv2.Canny(roi_smooth, 30, 100)
    gx = cv2.Sobel(roi_smooth, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(roi_smooth, cv2.CV_64F, 0, 1, ksize=5)
    
    # 4. Filter Edges by Orientation
    y_edge, x_edge = np.where(edges > 0)
    rx = x_edge - roi_center_x
    ry = y_edge - roi_center_y
    r_norm = np.sqrt(rx**2 + ry**2)
    r_norm[r_norm == 0] = 1
    
    g_x_val = gx[y_edge, x_edge]
    g_y_val = gy[y_edge, x_edge]
    g_norm = np.sqrt(g_x_val**2 + g_y_val**2)
    g_norm[g_norm == 0] = 1
    
    dot_product = np.abs((g_x_val/g_norm) * (rx/r_norm) + (g_y_val/g_norm) * (ry/r_norm))
    valid_indices = dot_product > 0.6
    
    clean_edges = np.zeros_like(edges)
    clean_edges[y_edge[valid_indices], x_edge[valid_indices]] = 255
    
    # 5. Distance Masking
    h, w = roi.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_guess = np.sqrt((X - roi_center_x)**2 + (Y - roi_center_y)**2)
    mask = (dist_from_guess > radius_estimate * 0.75) & (dist_from_guess < radius_estimate * 1.25)
    clean_edges[~mask] = 0
    
    if np.sum(clean_edges) < 10: return center_guess

    # 6. Hough Transform
    hough_radii = np.arange(int(radius_estimate * 0.8), int(radius_estimate * 1.2), 2)
    if len(hough_radii) == 0: return center_guess
    hough_res = hough_circle(clean_edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    
    if len(cx) == 0: return center_guess
    return (x_start + cx[0], y_start + cy[0])
    
def linear_profiles(channels_data, ves_coordinates, image_dim, parameters_profiles):
    """
    Calculation of the multiple linear profiles for a single vesicle.
    """
    num_channels   = len(channels_data[0,0,:])
    xc             = ves_coordinates[1]
    yc             = ves_coordinates[2]
    radius         = ves_coordinates[3]
    
    xc_refined, yc_refined = refine_guv_center(channels_data[:,:,0], (xc, yc), radius)
    
    ves_coordinates[1] = xc_refined
    ves_coordinates[2] = yc_refined
    
    num_angles     = int(parameters_profiles[0])
    length_excess  = parameters_profiles[1]
    dr             = parameters_profiles[2]
    
    profile_radius_limit = length_excess * radius
    
    theta        = np.linspace(0, 2*np.pi, num_angles, endpoint=False)         
    along_radius = np.arange(0, int(profile_radius_limit), dr)
    profile_radius = len(along_radius)
    
    intensity_profiles = np.zeros((profile_radius, num_angles, num_channels))
    
    death_mark = False
    if xc_refined + profile_radius_limit >= image_dim[0] or xc_refined - profile_radius_limit < 0 or \
       yc_refined + profile_radius_limit >= image_dim[1] or yc_refined - profile_radius_limit < 0:
        death_mark = True
        return intensity_profiles, along_radius, theta, death_mark
          
    x_coords = xc_refined + along_radius[np.newaxis, :] * np.cos(theta[:, np.newaxis])
    y_coords = yc_refined + along_radius[np.newaxis, :] * np.sin(theta[:, np.newaxis])
    
    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=0)
    
    for k in range(num_channels):
        channel = channels_data[:,:,k]
        profiles_flat = map_coordinates(channel, coords, order=1, mode='constant', cval=0.0)
        intensity_profiles[:,:,k] = profiles_flat.reshape(num_angles, profile_radius).T
             
    return intensity_profiles, along_radius, theta, death_mark
    

def fill_in_mask(mask_memb):
    """
    Assistive function that fills in the space inside the vesicle contours.
    """
    mask_memb_fill = nd.binary_fill_holes(mask_memb).astype(int)
    
    border_cuts_left = np.where(mask_memb[:,0] == 1)[0]
    
    if len(border_cuts_left[:]) > 2:
        assistant_left = np.zeros((len(mask_memb[:,0]),1))
        assistant_left[border_cuts_left[0]:border_cuts_left[-1]] = 1
        mask_left = np.hstack((assistant_left, mask_memb_fill))
        mask_fill_left = nd.binary_fill_holes(mask_left).astype(int)
        mask_memb_fill = mask_fill_left[:,1:]
       
    border_cuts_right = np.where(mask_memb[:,-1] == 1)[0]
    
    if len(border_cuts_right[:]) > 2:
        assistant_right = np.zeros((len(mask_memb[:,-1]),1))
        assistant_right[border_cuts_right[0]:border_cuts_right[-1]] = 1
        mask_right = np.hstack((mask_memb_fill, assistant_right))
        mask_fill_right = nd.binary_fill_holes(mask_right).astype(int)
        mask_memb_fill = mask_fill_right[:,:-1]
    
    border_cuts_up = np.where(mask_memb[0,:] == 1)[0]
    
    if len(border_cuts_up[:]) > 2:
        assistant_up = np.zeros((1,len(mask_memb[0,:])))
        assistant_up[border_cuts_up[0]:border_cuts_up[-1]] = 1
        mask_up = np.vstack((assistant_up, mask_memb_fill))
        mask_fill_up = nd.binary_fill_holes(mask_up).astype(int)
        mask_memb_fill = mask_fill_up[1:,:]
        
    border_cuts_down = np.where(mask_memb[-1,:] == 1)[0]
    
    if len(border_cuts_down[:]) > 2:
        assistant_down = np.zeros((1,len(mask_memb[0,:])))
        assistant_down[border_cuts_down[0]:border_cuts_down[-1]] = 1
        mask_down = np.vstack((mask_memb_fill, assistant_down))
        mask_fill_down = nd.binary_fill_holes(mask_down).astype(int)
        mask_memb_fill = mask_fill_down[:-1,:]
        
    return mask_memb_fill


def background_noise(plot_mask, channels_data, proteins_present, size_mask, image_dim, ves_coordinates, threshold, final_output_path, exp_info):
    """
    Calculation of the background noise.
    """
    num_channels     = len(channels_data[0,0,:])
    size_box = int(size_mask * ves_coordinates[3])

    vesicle_box = np.zeros((2*size_box, 2*size_box, num_channels))
    
    for i in range(num_channels):
        vesicle_box[:,:,i] = zoom_in_vesicle(channels_data[:,:,i], size_mask, image_dim, ves_coordinates)
        
    mask_memb = vesicle_box[:,:,0] > threshold
    mask_memb_fill = fill_in_mask(mask_memb)
    
    background = np.zeros((1, num_channels))
    
    for i in range(num_channels):
        background[:,i] = np.mean(np.ma.array(vesicle_box[:,:,i], mask=mask_memb_fill))
    
    if plot_mask == True:
        protein_status = plot_format(proteins_present)    
        if protein_status == "both_proteins":
            show_mask_both(int(ves_coordinates[0]), vesicle_box, mask_memb_fill, background, final_output_path, exp_info)
        elif protein_status == "only_septin":
            show_mask_only_sept(int(ves_coordinates[0]), vesicle_box, mask_memb_fill, background, final_output_path, exp_info)
        elif protein_status == "only_actin":
            show_mask_only_actin(int(ves_coordinates[0]), vesicle_box, mask_memb_fill, background, final_output_path, exp_info)
    
    return background


def plot_format(proteins_present):
    """
    Pre-processing for choosing a plotting format appropriate for the number of
    channels present.
    """
    if proteins_present[0] and proteins_present[1]: return "both_proteins"
    if proteins_present[0]: return "only_septin"
    if proteins_present[1]: return "only_actin"
    return "unknown"

            
def show_mask_both(vesicle_id, vesicle_box, mask_memb_fill, background, path_to_output, exp_info):
    """
    Displays the background noise calculated and the produced mask.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(13,4), dpi=125)
    fig.suptitle('Vesicle ' + str(vesicle_id) + ' :  Background:   M:' +  str(round(background[0,0],3)) + ', S:' +  str(round(background[0,1],3)) + ', A:' +  str(round(background[0,2],3)))
    
    ax1.imshow(vesicle_box[:,:,0], cmap = "cmap_cyan")
    ax1.set_title('Membrane')
    ax1.set_axis_off()
    
    ax2.imshow(vesicle_box[:,:,1] , cmap = "cmap_magenta_enhanced")
    ax2.set_title('Septin')
    ax2.set_axis_off()
    
    ax3.imshow(vesicle_box[:,:,2] , cmap = "cmap_yellow")
    ax3.set_title('Actin')
    ax3.set_axis_off()
    
    ax4.imshow(mask_memb_fill , cmap = "binary_r")
    ax4.set_title('Mask')
    ax4.set_axis_off()
    
    filename = f"Vesicle_{vesicle_id}_Channels_and_Mask.png"
    fig.savefig(os.path.join(path_to_output, filename))

    
def show_mask_only_sept(vesicle_id, vesicle_box, mask_memb_fill, background, path_to_output, exp_info):
   """
   Displays the background noise calculated and the produced mask (Septin only).
   """
   fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(13,5), dpi=125)
   fig.suptitle('Vesicle ' + str(vesicle_id) + ' :  Background:   M:' +  str(round(background[0,0],3)) + ', S:' +  str(round(background[0,1],3)))
   
   ax1.imshow(vesicle_box[:,:,0], cmap = "cmap_cyan")
   ax1.set_title('Membrane')
   ax1.set_axis_off()
   
   ax2.imshow(vesicle_box[:,:,1] , cmap = "cmap_magenta_enhanced")
   ax2.set_title('Septin')
   ax2.set_axis_off()
   
   ax3.imshow(mask_memb_fill , cmap = "binary_r")
   ax3.set_title('Mask')
   ax3.set_axis_off()
   
   filename = f"Vesicle_{vesicle_id}-Channels_and_Mask.png"
   fig.savefig(os.path.join(path_to_output, filename))


def show_mask_only_actin(vesicle_id, vesicle_box, mask_memb_fill, background, path_to_output, exp_info):
   """
   Displays the background noise calculated and the produced mask (Actin only).
   """
   fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(13,5), dpi=125)
   fig.suptitle('Vesicle ' + str(vesicle_id) + ' :  Background:   M:' +  str(round(background[0,0],3)) + ', A:' +  str(round(background[0,1],3))) 
   
   ax1.imshow(vesicle_box[:,:,0], cmap = "cmap_cyan")
   ax1.set_title('Membrane')
   ax1.set_axis_off()
   
   ax2.imshow(vesicle_box[:,:,1] , cmap = "cmap_yellow")
   ax2.set_title('Actin')
   ax2.set_axis_off()
   
   ax3.imshow(mask_memb_fill , cmap = "binary_r")
   ax3.set_title('Mask')
   ax3.set_axis_off()
   
   filename = f"Vesicle_{vesicle_id}-Channels_and_Mask.png"
   fig.savefig(os.path.join(path_to_output, filename))


def background_correction(num_channels, intensity_profiles, background):
    """ 
    Background correction of the intesity linear profiles. 
    """
    intensity_profiles_corrected = np.ones_like(intensity_profiles)
    
    for i in range(num_channels):
        intensity_profiles_corrected[:,:,i] = np.clip(intensity_profiles[:,:,i] - background[:,i], 0, None) # setting negative values occuring after subtraction to 0

    return intensity_profiles_corrected


def radial_profile(num_channels, intensity_profiles):
    """
    Calculation of a collective radial profile.
    """
    radial_profiles  = np.ones_like(intensity_profiles[:,0,:])
    
    for i in range(num_channels):
        radial_profiles[:,i]  = np.average(intensity_profiles[:,:,i], axis = 1)
         
    return radial_profiles


def membrane_detection(radial_profile_memb, radius):
    """
    Detection of the membrane inner and outer borders.
    UPDATED: Returns 'peak_index' for refined radius calculation.
    """
    comment = []
    death_mark = False
    peak_index = 0 # NEW
    
    # Find peaks
    peaks, properties = find_peaks(radial_profile_memb, 
                                   height=np.max(radial_profile_memb) * 0.1, 
                                   distance=5, 
                                   prominence=np.max(radial_profile_memb) * 0.05)
    
    if not np.any(peaks):
        return 0, 0, 0, ["no_memb_peak"], True # Return peak_index=0
    
    # Peak Selection Logic
    if len(peaks) == 1:
        chosen_peak = 0
    else:
        peak_scores = []
        for i, peak_pos in enumerate(peaks):
            dist_score = 1.0 / (1.0 + abs(peak_pos - radius) / radius)
            height_score = radial_profile_memb[peak_pos] / np.max(radial_profile_memb)
            score = dist_score * 3.0 + height_score * 2.0
            peak_scores.append(score)
        chosen_peak = np.argmax(peak_scores)
    
    peak_index = peaks[chosen_peak] # Capture peak index
    

    try:
        # Measure width at 75% height (lower down the peak) to capture tails
        # Standard FWHM is 0.5; using 0.75 captures more of the base
        # Note: peak_widths returns (widths, width_heights, left_ips, right_ips)
        width_results = peak_widths(radial_profile_memb, [peak_index], rel_height=0.75)
        
        # Get interpolated indices
        left_idx, right_idx = width_results[2][0], width_results[3][0]
        
        # Add PADDING to ensure we don't cut off signal
        # This prevents "leaking" signal into the background calculation
        padding = 3  # pixels
        index_border_in = int(np.floor(left_idx - padding))
        index_border_out = int(np.ceil(right_idx + padding))
        
        # Safety clamp
        index_border_in = max(0, index_border_in)
        index_border_out = min(len(radial_profile_memb)-1, index_border_out)
        
    except:
        index_border_in = max(0, peaks[chosen_peak] - 8)
        index_border_out = min(len(radial_profile_memb) - 1, peaks[chosen_peak] + 8)
        comment.append("width_calc_failed")

    
    # Quality checks
    if index_border_out - index_border_in > radius/3: 
        comment.append("wide_peak")
    
    return index_border_in, index_border_out, peak_index, comment, death_mark

def check_membrane_quality(angular_profiles, radial_profiles, radius):
    """
    Check if the membrane is uniform enough for reliable analysis.
    Flags problematic vesicles for manual review but doesn't reject them.
    """
    comments = []
    
    # Check angular uniformity of membrane signal
    membrane_angular = angular_profiles[:,0]  # Membrane channel
    
    # Calculate coefficient of variation (CV) - standard deviation / mean
    if np.mean(membrane_angular) > 0:  # Avoid division by zero
        cv_membrane = np.std(membrane_angular) / np.mean(membrane_angular)
        
        # Flag irregular membranes for review
        if np.mean(membrane_angular) > 0:
            cv_membrane = np.std(membrane_angular) / np.mean(membrane_angular)
            if cv_membrane > 0.4: comments.append("very_irregular_membrane")
            elif cv_membrane > 0.3: comments.append("irregular_membrane")
    
    # CHECK 1: Look for sharp transitions
    if len(membrane_angular) > 10:
        # Calculate differences between adjacent points
        membrane_diff = np.abs(np.diff(membrane_angular))
        max_transition = np.max(membrane_diff)
        mean_signal = np.mean(membrane_angular)
        
        # Flag if there are very sharp transitions
        if max_transition > mean_signal * 0.5:  # Transition > 50% of mean signal
            comments.append("sharp_membrane_transitions")
    
    # CHECK 2: Look for plateau-like behavior (flat regions)
    if len(membrane_angular) > 20:
        # Check for long flat regions
        rolling_std = []
        window_size = 20  # Check 20-degree windows
        
        for i in range(len(membrane_angular) - window_size):
            window_std = np.std(membrane_angular[i:i+window_size])
            rolling_std.append(window_std)
        
        min_rolling_std = np.min(rolling_std) if rolling_std else 0
        max_rolling_std = np.max(rolling_std) if rolling_std else 0
        
        # If there are very flat regions alongside variable regions
        if min_rolling_std < 5 and max_rolling_std > 15:
            comments.append("membrane_plateau_regions")
    
    # CHECK 3: Check for protein clustering that might affect localization
    if angular_profiles.shape[1] > 1:  # If we have protein channels
        for protein_idx in range(1, angular_profiles.shape[1]):
            protein_angular = angular_profiles[:, protein_idx]
            
            if np.mean(protein_angular) > 5:  # Only check if there's significant signal
                protein_cv = np.std(protein_angular) / np.mean(protein_angular)
                
                # Flag high protein clustering
                if protein_cv > 0.8:
                    comments.append("high_protein_clustering")
                elif protein_cv > 0.6:
                    comments.append("moderate_protein_clustering")
    
    # CHECK 4: Check for asymmetric membrane profiles
    if len(membrane_angular) >= 180:  # Need enough points to check symmetry
        # Compare first half vs second half
        first_half = membrane_angular[:len(membrane_angular)//2]
        second_half = membrane_angular[len(membrane_angular)//2:]
        
        # Make them the same length for comparison
        min_len = min(len(first_half), len(second_half))
        first_half = first_half[:min_len]
        second_half = second_half[:min_len]
        
        # Calculate difference between halves
        if np.mean(first_half) > 0 and np.mean(second_half) > 0:
            asymmetry_ratio = abs(np.mean(first_half) - np.mean(second_half)) / max(np.mean(first_half), np.mean(second_half))
            
            if asymmetry_ratio > 0.3:  # More than 30% difference between halves
                comments.append("asymmetric_membrane")
    
    # Existing checks for spikes, valleys, etc...
    if len(membrane_angular) > 0:
        membrane_median = np.median(membrane_angular)
        max_spike = np.max(membrane_angular)
        min_valley = np.min(membrane_angular)
        
        if max_spike > membrane_median * 3:
            comments.append("membrane_spikes")
        
        if membrane_median > 0 and min_valley < membrane_median * 0.3:
            comments.append("membrane_valleys")
    
    # Existing signal strength checks...
    mean_membrane = np.mean(membrane_angular)
    if mean_membrane < 10:
        comments.append("weak_membrane_signal")
    elif mean_membrane < 20:
        comments.append("low_membrane_signal")
    
    return comments


def angular_profile(num_channels, intensity_profiles, index_border_in, index_border_out, pixels_to_remove):
    """
    Calculation of the angular profile along the membrane contour.
    
    UPDATED: Uses np.max instead of np.average. 
    This prevents wide border detection from diluting the intensity signal,
    ensuring localization is calculated based on peak protein density.
    """
    # Initialize array with shape (num_angles, num_channels)
    angular_profiles = np.ones_like(intensity_profiles[0,:,:])
    
    for i in range(num_channels):
        # Slice the radial profile to getting only the segment within the borders
        # Shape of segment: (border_width, num_angles)
        segment = intensity_profiles[pixels_to_remove+index_border_in : pixels_to_remove+index_border_out, :, i]
        
        # Calculate the MAXIMUM intensity found along the radius for this angle
        # axis=0 collapses the radial dimension, leaving us with one value per angle
        angular_profiles[:,i] = np.max(segment, axis=0)
    
    return angular_profiles


def localization(num_channels, angular_profiles, radial_profiles, index_border_in, index_border_out, radius, size_central_area, pixels_to_remove):
    """
    Quantification of the protein localization on the membrane.
    FIXED: Prevents divide-by-zero errors when median intensity is 0.
    """
    comment = []
    num_proteins = num_channels - 1 
    index_centre = int(size_central_area * (radius-pixels_to_remove)) 
    localization = np.zeros(num_proteins)
 
    for i in range(num_proteins):
        if np.mean(angular_profiles[:,i+1]) > 0:
            rsd = np.std(angular_profiles[:,i+1]) / np.mean(angular_profiles[:,i+1])
            if rsd > 0.8:
                comment = ["high_rsd"]
        
        median = np.median(angular_profiles[:,i+1])
        centre = np.average(radial_profiles[0:index_centre,i+1]) 
        
        # FIX: Check if median is valid to avoid RuntimeWarning: divide by zero
        if median > 0.001:
            localization[i] = np.clip((median - centre)/median, 0, None)
        else:
            localization[i] = 0.0 # No signal on membrane = No localization
    
    return localization, comment


def analyze_actin_structure(radial_profiles, angular_profiles, localization_score, protein_channel_index, px_size=1.0, border_in=0, border_out=0):
    """
    Calculates actin cortex properties.
    FIXED: Calculates Density (rho) for ALL vesicles (including Lumenal Actin).
    Only calculates Thickness (t_cortex) if localization is significant AND peak is at border.
    """
    # Initialize defaults
    t_cortex, rho_actin, uniformity = 0.0, 0.0, 0.0
    
    # --- 1. ALWAYS Calculate Density (rho_actin) ---
    # Density exists regardless of where the actin is (lumen or cortex)
    actin_angular = angular_profiles[:, protein_channel_index]
    rho_actin = np.mean(actin_angular)
    
    # --- 2. Calculate Uniformity (CV) ---
    if rho_actin > 0:
        stdev = np.std(actin_angular)
        uniformity = stdev / rho_actin

    # --- 3. Conditionally Calculate Thickness (t_cortex) ---
    # Only try to measure thickness if there is actually a cortex (Localization > 0.2)
    if localization_score >= 0.2:
        
        actin_radial = radial_profiles[:, protein_channel_index]
        
        # Find peaks in the actin channel
        peaks, properties = find_peaks(actin_radial, height=np.max(actin_radial)*0.5)
        
        if len(peaks) > 0:
            # Find the tallest peak
            tallest_peak_idx = np.argmax(actin_radial[peaks])
            peak_pos = peaks[tallest_peak_idx]
            
            # SPATIAL CHECK: Is the peak inside the membrane region?
            # We allow a small buffer (e.g. +/- 2 pixels)
            if (peak_pos >= border_in - 2) and (peak_pos <= border_out + 2):
                
                # Calculate FWHM (rel_height=0.5)
                widths, width_heights, left_ips, right_ips = peak_widths(actin_radial, peaks, rel_height=0.5)
                
                # Get width of the tallest peak
                t_cortex_px = widths[tallest_peak_idx]
                t_cortex = t_cortex_px * px_size

    return t_cortex, rho_actin, uniformity


def plot_debug_overlay(ves_coordinates, radius, index_border_in, index_border_out, 
                       image_dim, channels_data, along_radius, theta, 
                       final_output_path):
    """
    Plots a debug image showing radial spokes and borders.
    """
    xc, yc = ves_coordinates[1], ves_coordinates[2]
    
    # Create zoom box
    size_view = 1.5  # View 1.5x radius
    membrane_channel = channels_data[:,:,0]
    vesicle_img = zoom_in_vesicle(membrane_channel, size_view, image_dim, ves_coordinates)
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(vesicle_img, cmap='gray')
    ax.set_title(f"Debug: Vesicle {int(ves_coordinates[0])}")
    
    # Image center in zoomed coordinates
    center_img = vesicle_img.shape[0] // 2
    
    # 1. Plot Radial Spokes (Yellow) - Plot every 15th line to avoid clutter
    for angle in theta[::15]:
        r_max = along_radius[-1]
        x_end = center_img + r_max * np.cos(angle)
        y_end = center_img + r_max * np.sin(angle)
        ax.plot([center_img, x_end], [center_img, y_end], color='yellow', alpha=0.3, lw=0.5)

    # 2. Plot Detected Borders (Red/Orange Circles)
    radius_in = along_radius[index_border_in]
    radius_out = along_radius[index_border_out]
    
    circ_in = plt.Circle((center_img, center_img), radius_in, color='red', fill=False, lw=1.5, label='Inner Border')
    circ_out = plt.Circle((center_img, center_img), radius_out, color='orange', fill=False, lw=1.5, label='Outer Border')
    ax.add_patch(circ_in)
    ax.add_patch(circ_out)
    
    # 3. Add Legend
    ax.legend(loc='upper right', fontsize='small')
    ax.axis('off')
    
    fig.savefig(os.path.join(final_output_path, f"Vesicle_{int(ves_coordinates[0])}_DEBUG.png"))
    plt.close(fig)


def plot_intensity_profiles(plot_intensity_profiles, proteins_present, parameters_sizes, channels_data, ves_coordinates, image_dim, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, background, localization, path_to_output, exp_info, pixel_size):
    """
    Optional plotting of the calculated intensity radial and angular profiles.
    UPDATED: Now passes pixel_size to plotting functions.
    """
    if plot_intensity_profiles == True:
        
        num_channels     = len(channels_data[0,0,:])
        size_box = int(parameters_sizes[1] * ves_coordinates[3])

        image_box = np.zeros((2*size_box, 2*size_box, num_channels))
    
        for i in range(num_channels):
            image_box[:,:,i] = zoom_in_vesicle(channels_data[:,:,i], parameters_sizes[1], image_dim, ves_coordinates)
    
        protein_status = plot_format(proteins_present)    

        if protein_status == "both_proteins":
            plot_int_prof_both(int(ves_coordinates[0]), ves_coordinates[3], parameters_sizes[2], along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info, pixel_size)
    
        elif protein_status == "only_septin":
            plot_int_prof_only_septin(int(ves_coordinates[0]), ves_coordinates[3], parameters_sizes[2], along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info, pixel_size)
        
        elif protein_status == "only_actin":
            plot_int_prof_only_actin(int(ves_coordinates[0]), ves_coordinates[3], parameters_sizes[2], along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info, pixel_size)

def plot_int_prof_both(vesicle_id, radius, size_central_area, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info, pixel_size):
    """
    Plotting of the intensity radial and angular profiles (Both proteins).
    UPDATED: Uses physical units (um) and visualizes FWHM.
    """
    radius_um = along_radius * pixel_size
    
    ## Intensity radial profile:
    fig, axes = plt.subplot_mosaic("AB;AC;AD;AE", width_ratios = [3, 1], dpi=125)
    axes["A"].set_title('Intensity radial profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(radius_um, radial_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(radius_um, radial_profiles[:,1], c = 'magenta', label = 'septin')
    axes["A"].plot(radius_um, radial_profiles[:,2], c = 'gold', label = 'actin')
    
    # Visualizing FWHM
    actin_prof = radial_profiles[:, 2] 
    peaks, _ = find_peaks(actin_prof, height=np.max(actin_prof)*0.5)
    if len(peaks) > 0:
        widths, width_heights, left_ips, right_ips = peak_widths(actin_prof, peaks, rel_height=0.5)
        #idx = np.argmax(widths)
        idx = np.argmax(actin_prof[peaks])
        r_left_um = (along_radius[0] + left_ips[idx]) * pixel_size
        r_right_um = (along_radius[0] + right_ips[idx]) * pixel_size
        axes["A"].hlines(width_heights[idx], r_left_um, r_right_um, color='blue', lw=2, label='Actin FWHM')

    axes["A"].axvline(x = along_radius[index_border_in]*pixel_size, c = 'red', label = 'border in', ls="--")
    axes["A"].axvline(x = along_radius[index_border_out]*pixel_size, c = 'orange', label = 'border out', ls="--")
    axes["A"].set_xlabel('radius (µm)')
    axes["A"].set_ylabel('I (a.u.)')
    axes["A"].legend(loc="upper left", fontsize=8)
    
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan"); axes["B"].set_axis_off()
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced"); axes["C"].set_axis_off()
    axes["D"].imshow(image_box[:,:,2], cmap = "cmap_yellow"); axes["D"].set_axis_off()
    
    axes["E"].text(-0.2,0.7,"Background M:  " + str(round(background[0,0],4)), size = 10)
    axes["E"].text(-0.2,0.5,"Background S:   " + str(round(background[0,1],4)), size = 10)
    axes["E"].text(-0.2,0.3,"Background A:   " + str(round(background[0,2],4)), size = 10)
    axes["E"].text(-0.2,0.0,"Localization S :  " + str(round(localization[0],3)), size = 10)
    axes["E"].text(-0.2,-0.2,"Localization A :  " + str(round(localization[1],3)), size = 10)
    axes["E"].set_axis_off()
    
    filename = f"Vesicle_{vesicle_id}-Int_prof_Radial.png"
    fig.savefig(os.path.join(path_to_output, filename))
    
    ## Intensity angular profile:
    fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1], dpi=125)
    axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,1], c = 'magenta', label = 'septin')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,2], c = 'gold', label = 'actin')
    axes["A"].legend(loc="upper right")
    axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'I (a.u.)')
    
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan"); axes["B"].set_axis_off()
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced"); axes["C"].set_axis_off()
    axes["D"].imshow(image_box[:,:,2], cmap = "cmap_yellow"); axes["D"].set_axis_off()
    
    filename = f"Vesicle_{vesicle_id}-Int_prof_Angular.png"
    fig.savefig(os.path.join(path_to_output, filename))
    plt.close()

    
def plot_int_prof_only_septin(vesicle_id, radius, size_central_area, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info, pixel_size):
    """
    Plotting of the intensity radial and angular profiles (Septin only).
    """
    radius_um = along_radius * pixel_size

    ## Intensity radial profile:
    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [6, 2], dpi=125)
    axes["A"].set_title('Intensity radial profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(radius_um[2:], radial_profiles[2:,0]/np.max(radial_profiles[2:,0]), c = 'cyan', label = 'Membrane channel')
    axes["A"].plot(radius_um[2:], radial_profiles[2:,1]/np.max(radial_profiles[2:,1]), c = 'magenta', label = 'Septin channel')
    axes["A"].set_xlabel('radius (µm)')
    axes["A"].set_ylabel('normalized intensity (a.u.)')
    
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan"); axes["B"].set_axis_off()
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced"); axes["C"].set_axis_off()
    
    filename = f"Vesicle_{vesicle_id}-Int_prof_Radial.png"
    fig.savefig(os.path.join(path_to_output, filename))
    
    ## Intensity angular profile:
    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [6, 2], dpi=125)
    axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,0]/max(angular_profiles[:,0]), c = 'cyan', label = 'Membrane channel')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,1]/max(angular_profiles[:,1]), c = 'magenta', label = 'Septin channel')
    axes["A"].legend(loc="upper right", fontsize=10)
    axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'normalized intensity (a.u.)')
    
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan"); axes["B"].set_axis_off()
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced"); axes["C"].set_axis_off()
    
    filename = f"Vesicle_{vesicle_id}-Int_prof_Angular.png"
    fig.savefig(os.path.join(path_to_output, filename))
    plt.close()
        
    
def plot_int_prof_only_actin(vesicle_id, radius, size_central_area, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info, pixel_size):
    """
    Plotting of the intensity radial and angular profiles (Actin only).
    """
    radius_um = along_radius * pixel_size
    
    ## Intensity radial profile:
    fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1], dpi=125)
    axes["A"].set_title('Intensity radial profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(radius_um, radial_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(radius_um, radial_profiles[:,1], c = 'gold', label = 'actin')
    
    actin_prof = radial_profiles[:, 1] 
    peaks, _ = find_peaks(actin_prof, height=np.max(actin_prof)*0.5)
    if len(peaks) > 0:
        widths, width_heights, left_ips, right_ips = peak_widths(actin_prof, peaks, rel_height=0.5)
        #idx = np.argmax(widths)
        idx = np.argmax(actin_prof[peaks])
        r_left_um = (along_radius[0] + left_ips[idx]) * pixel_size
        r_right_um = (along_radius[0] + right_ips[idx]) * pixel_size
        axes["A"].hlines(width_heights[idx], r_left_um, r_right_um, color='blue', lw=2, label='Actin FWHM')

    axes["A"].axvline(x = along_radius[index_border_in]*pixel_size, c = 'red', label = 'border in', ls="--")
    axes["A"].axvline(x = along_radius[index_border_out]*pixel_size, c = 'orange', label = 'border out', ls="--")
    axes["A"].set_xlabel('radius (µm)')
    axes["A"].set_ylabel('I (a.u.)')
    axes["A"].legend(loc="upper left", fontsize=8)
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan"); axes["B"].set_axis_off()
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_yellow"); axes["C"].set_axis_off()
    axes["D"].text(-0.2,0.7,"Background M:  " + str(round(background[0,0],4)), size = 10)
    axes["D"].text(-0.2,0.5,"Background A:   " + str(round(background[0,1],4)), size = 10)
    axes["D"].text(-0.2,0.0,"Localization A :  " + str(round(localization[0],3)), size = 10)
    axes["D"].set_axis_off()
    
    filename = f"Vesicle_{vesicle_id}-Int_prof_Radial.png"
    fig.savefig(os.path.join(path_to_output, filename))
    
    ## Intensity angular profile:
    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [3, 1], dpi=125)
    axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,1], c = 'gold', label = 'actin')
    axes["A"].legend(loc="upper left")
    axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'I (a.u.)')
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan"); axes["B"].set_axis_off()
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_yellow"); axes["C"].set_axis_off()

    filename = f"Vesicle_{vesicle_id}-Int_prof_Angular.png"
    fig.savefig(os.path.join(path_to_output, filename))
    plt.close()

def plot_size_distribution(refined_radii_um, path_to_output):
    """
    NEW: Plots a histogram of the refined vesicle radii in microns.
    """
    if len(refined_radii_um) == 0:
        return

    plt.figure(figsize=(6, 4), dpi=125)
    plt.hist(refined_radii_um, bins=10, color='gray', edgecolor='black', alpha=0.7)
    plt.title(f"Size Distribution (N={len(refined_radii_um)})")
    plt.xlabel("Refined Radius (µm)")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(path_to_output, "Size_Distribution_Refined.png"))
    plt.close()

def format_result_row(exp_info, ves_coordinates, background, localization, t_cortex, rho_actin, uniformity, refined_radius_um, comment):
    """
    Prepares the data row for the CSV file.
    Does NOT write to file.
    """
    list_exp_info        = list(exp_info[:-1])
    list_ves_coordinates = list(ves_coordinates)
    list_background      = list(background[0,:])
    list_localization    = list(localization)

    # Create list of structure metrics (Handle if None)
    if t_cortex is not None:
        list_structure = [round(float(t_cortex), 3), round(float(rho_actin), 3), round(float(uniformity), 3)]
    else:
        list_structure = [] 
        
    # Handle refined radius
    radius_val = [round(float(refined_radius_um), 3)] if refined_radius_um else [0]

    vesicle_row = list_exp_info + list_ves_coordinates + list_background + list_localization + list_structure + radius_val + comment
    return vesicle_row

def edit_output(final_output_path, exp_info, ves_coordinates, background, localization, t_cortex, rho_actin, uniformity, refined_radius_um, comment):
    """
    Original function maintained for backward compatibility.
    Calls format_result_row then writes to file immediately.
    """
    row = format_result_row(exp_info, ves_coordinates, background, localization, t_cortex, rho_actin, uniformity, refined_radius_um, comment)
    with open(os.path.join(final_output_path, "Analysis_Results.csv"), "a", newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(row)

import warnings # Add this import at the top of skeleton.py if not present

def process_single_vesicle(ves_coordinates, channels_data, image_dim, parameters_profiles, 
                           proteins_present, size_mask, threshold_membrane, final_output_path, 
                           exp_info, parameters_sizes, plot_int_profiles, plot_mask, pixel_size):
    """
    Runs the entire analysis pipeline for a single vesicle.
    Designed for parallel processing. Returns the result row instead of writing it.
    """
    # Suppress peak warnings and register colormaps
    warnings.filterwarnings("ignore", category=UserWarning) 
    # specific warning from scipy signal
    warnings.filterwarnings("ignore", message="some peaks have a width of 0")
    color_maps()
    
    num_channels = channels_data.shape[2]
    comment = []
    
    # 1. Linear profiles
    intensity_profiles, along_radius, theta, death_mark = linear_profiles(channels_data, ves_coordinates, image_dim, parameters_profiles)
    
    if death_mark:
        background = np.zeros((1, num_channels))
        localization_val = np.zeros(num_channels-1)
        comment = ["margins"]
        row = format_result_row(exp_info, ves_coordinates, background, localization_val, None, None, None, 0, comment)
        return row, None

    # 2. Background
    background = background_noise(plot_mask, channels_data, proteins_present, size_mask, image_dim, ves_coordinates, threshold_membrane, final_output_path, exp_info)
    intensity_profiles_corrected = background_correction(num_channels, intensity_profiles, background)
    
    # 3. Radial Profile
    radial_profiles = radial_profile(num_channels, intensity_profiles_corrected)
    pixels_to_remove = 2
    radial_profiles = radial_profiles[pixels_to_remove:,:]
    along_radius = along_radius[pixels_to_remove:]
    
    # 4. Membrane Detection
    index_border_in, index_border_out, peak_index, comment_peak, death_mark_peak = membrane_detection(radial_profiles[:,0], ves_coordinates[3])
    
    if comment_peak: comment += comment_peak
    
    refined_radius_px = along_radius[peak_index]
    refined_radius_um = refined_radius_px * pixel_size
    
    # Debug Plot 
    plot_debug_overlay(ves_coordinates, ves_coordinates[3], index_border_in, index_border_out, image_dim, channels_data, along_radius, theta, final_output_path)
    
    if death_mark_peak:
         background = np.zeros((1, num_channels))
         localization_val = np.zeros(num_channels-1)
         comment = comment_peak
         row = format_result_row(exp_info, ves_coordinates, background, localization_val, None, None, None, refined_radius_um, comment)
         plt.close('all') # Cleanup
         return row, refined_radius_um

    # Low signal check
    for j in range(num_channels-1):
        if max(radial_profiles[:index_border_out, j+1]) <= 5:
            comment.append("low_protein")
            
    # 5. Angular Profile
    angular_profiles = angular_profile(num_channels, intensity_profiles_corrected, index_border_in, index_border_out, pixels_to_remove)
    
    # Quality Check
    quality_comments = check_membrane_quality(angular_profiles, radial_profiles, ves_coordinates[3])
    if quality_comments: comment += quality_comments
    
    if index_border_in == index_border_out:
        comment.append("no_memb_detected")
        comment_str = [', '.join(comment)]
        background = np.zeros((1, num_channels))
        localization_val = np.zeros(num_channels-1)
        row = format_result_row(exp_info, ves_coordinates, background, localization_val, None, None, None, refined_radius_um, comment_str)
        plt.close('all') # Cleanup
        return row, refined_radius_um

    # 6. Localization
    size_central_area = parameters_sizes[2] 
    localization_val, comment_loc = localization(num_channels, angular_profiles, radial_profiles, index_border_in, index_border_out, ves_coordinates[3], size_central_area, pixels_to_remove)
    
    # 7. Actin Structure
    actin_idx = num_channels - 1
    loc_actin_score = localization_val[-1]
    
    # FIX: Pass the border indices to analyze_actin_structure
    t_cortex, rho_actin, uniformity = analyze_actin_structure(radial_profiles, angular_profiles, loc_actin_score, actin_idx, pixel_size, index_border_in, index_border_out)
    
    # --- FILTER EMPTY VESICLES (DISABLED) ---
    # We disable this so we can distinguish Empty vs Lumenal downstream
    # if rho_actin < 2.0:
    #     comment.append("empty_vesicle")
    #     localization_val[-1] = 0.0
    # ----------------------------------

    # Final comments
    if np.isnan(localization_val).any(): comment.append("no_memb_detected")
    elif np.isinf(localization_val).any(): comment.append("zero_at_centre")
    if comment_loc: comment += comment_loc
    
    # Plotting
    plot_intensity_profiles(plot_int_profiles, proteins_present, parameters_sizes, channels_data, ves_coordinates, image_dim, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, background, localization_val, final_output_path, exp_info, pixel_size)
    
    # FIX: Force close all figures created in this thread to prevent memory warning
    plt.close('all')

    # Format Row
    comment_final = [', '.join(comment)] if comment else ["OK"]
    row = format_result_row(exp_info, ves_coordinates, background, localization_val, t_cortex, rho_actin, uniformity, refined_radius_um, comment_final)
    
    return row, refined_radius_um