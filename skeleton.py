# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:33:44 2022

@author: kkourkoulou

Script to calculate the radial and angular intensity profiles of GUVs 
encapsulating septin and/or actin and to quantify the corresponding protein
localization on the vesicle membrane. 
 
main.py     :    main file to run the script
skeleton.py :    file with all necessary functions in main.py
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
from scipy import ndimage as nd                                                
from scipy.signal import find_peaks, peak_widths                             


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


def read_files(path_membrane, path_septin, path_actin, path_detected, exp_info, proteins_present):
    """
    Function that reads the image files and the vesicle detection from separate folders.
    """
    file_stem = exp_info[3]
    
    # 1. Read Membrane (C1) - Always present
    c1_path = os.path.join(path_membrane, file_stem + "-C1.tif")
    channels_data = plt.imread(c1_path)
    
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
    
    # Print data for debugging
    print(f"Processing: {file_stem}")
    
    # Create coordinates array 
    coordinates = np.zeros((len(detected_vesicles), 4))
    
    # ID, xc, yc, radius
    coordinates[:, 0] = np.arange(1, len(detected_vesicles) + 1)
    coordinates[:, 1] = detected_vesicles.iloc[:, 0].values  # X coordinate
    coordinates[:, 2] = detected_vesicles.iloc[:, 1].values  # Y coordinate  
    coordinates[:, 3] = detected_vesicles.iloc[:, 2].values  # Radius/size
    
    return channels_data, coordinates


def create_output(path_to_output, exp_info, proteins_present):
    """
    Creates an empty output directory and an output file.
    """
    os.makedirs(os.path.dirname(path_to_output), exist_ok= True)
        
    protein_status = plot_format(proteins_present)
    
    if protein_status == "both_proteins":
        column_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "A Background", "S localization", "A localization", "Comment"]

    if protein_status == "only_septin":
        column_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "S localization", "Comment"]

    if protein_status == "only_actin":
        column_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "A Background", "A localization", "Comment"]

    with open(path_to_output + "-Output.csv", "w", newline='') as output_file:
        df = csv.DictWriter(output_file, delimiter=',', fieldnames = column_headers)
        df.writeheader()
        
    
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

    
def detected_centres(to_plot, num_vesicles, ch_membrane, all_xc, all_yc, path_to_output, exp_info):
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
        
        plt.savefig(path_to_output + "-Detected_centres.png")


def zoom_in_vesicle(channel, size_side, image_dim, ves_coordinates):
    """
    Creates a square cropped view of a chosen vesicle. 
    Robustly handles cases where the crop box is larger than the image dimensions.
    """
    xc               = ves_coordinates[1]
    yc               = ves_coordinates[2]
    radius           = ves_coordinates[3]
    
    vesicle_box_side = int(size_side * radius)
    expected_size    = 2 * vesicle_box_side
    
    move_right = 0
    move_left  = 0
    move_up    = 0
    move_down  = 0

    # Calculate shifts to keep the window within image bounds if possible
    if xc - vesicle_box_side < 0: 
        move_right = vesicle_box_side - xc
                  
    if xc + vesicle_box_side > image_dim[0]: 
        move_left  = vesicle_box_side - (image_dim[0]-xc)   
        
    if yc - vesicle_box_side < 0: 
        move_down  = vesicle_box_side - yc           
                      
    if yc + vesicle_box_side > image_dim[1]: 
        move_up    = vesicle_box_side - (image_dim[1]-yc)   
    
    # Define slice indices
    y_start = int(yc - vesicle_box_side + move_down - move_up)
    y_end   = int(yc + vesicle_box_side + move_down - move_up)
    x_start = int(xc - vesicle_box_side + move_right - move_left)
    x_end   = int(xc + vesicle_box_side - move_left + move_right)
    
    # Perform the slice
    # Note: Python slices automatically clip to valid ranges (e.g. 0 to 2048)
    vesicle_box_channel = channel[max(0, y_start):max(0, y_end), max(0, x_start):max(0, x_end)]
    
    # Check if the resulting slice matches the expected size
    if vesicle_box_channel.shape != (expected_size, expected_size):
        # Create a full-size black canvas
        padded_box = np.zeros((expected_size, expected_size), dtype=channel.dtype)
        
        # Paste the valid image data into the canvas
        # We align it based on the effective valid data size we retrieved
        h, w = vesicle_box_channel.shape
        padded_box[:h, :w] = vesicle_box_channel
        
        return padded_box
    
    return vesicle_box_channel


def define_threshold(user_choice, channel, size_side, image_dim, ves_coordinates):    
    """
    Allows for optional manual choice of the threshold method.
    """
    vesicle_box_channel = zoom_in_vesicle(channel, size_side, image_dim, ves_coordinates)
    
    if user_choice == True:
        
        fig, ax = try_all_threshold(vesicle_box_channel, figsize=(5, 10), verbose=False)
        plt.show()                  

        threshold_method = input("Choose threshold method to follow:")
        
        if threshold_method == 'Isodata':
            threshold = threshold_isodata(vesicle_box_channel)
        elif threshold_method == 'Li':
            threshold = threshold_li(vesicle_box_channel)
        elif threshold_method == 'Mean':
            threshold = threshold_mean(vesicle_box_channel)
        elif threshold_method == 'Minimum':
            threshold = threshold_minimum(vesicle_box_channel)
        elif threshold_method == 'Otsu':
            threshold = threshold_otsu(vesicle_box_channel)
        elif threshold_method == 'Triangle':
            threshold = threshold_triangle(vesicle_box_channel)
        elif threshold_method == 'Yen':
            threshold = threshold_yen(vesicle_box_channel)
        else:
            print('Invalid method!')
    else:
        threshold = threshold_li(vesicle_box_channel)
        
    return threshold

    
def linear_profiles(channels_data, ves_coordinates, image_dim, parameters_profiles):
    """
    Calculation of the multiple linear profiles for a single vesicle.
    """
    num_channels   = len(channels_data[0,0,:])
    
    xc             = ves_coordinates[1]
    yc             = ves_coordinates[2]
    radius         = ves_coordinates[3]
    
    num_angles     = int(parameters_profiles[0])
    length_excess  = parameters_profiles[1]
    dr             = parameters_profiles[2]
    
    profile_radius_limit = length_excess * radius
    
    theta        = np.linspace(0, 2*np.pi, num_angles, endpoint=False)         
    along_radius = np.arange(0, int(profile_radius_limit), dr)
    profile_radius = len(along_radius)
    
    intensity_profiles = np.zeros((profile_radius, num_angles, num_channels))
    
    death_mark = False
    if xc + profile_radius_limit >= image_dim[0] or xc - profile_radius_limit < 0 or yc + profile_radius_limit >= image_dim[1] or yc - profile_radius_limit < 0:
        death_mark = True
        return intensity_profiles, along_radius, theta, death_mark
          
    line_x       = np.full((profile_radius, num_angles), int(xc)) + np.rint(np.matmul(np.transpose(np.asmatrix(along_radius)),np.asmatrix(np.cos(theta))))
    line_y       = np.full((profile_radius, num_angles), int(yc)) + np.rint(np.matmul(np.transpose(np.asmatrix(along_radius)),np.asmatrix(np.sin(theta))))
    
    for i in range(profile_radius):
        for j in range((num_angles)): 
            for k in range(num_channels):
                    channel = channels_data[:,:,k]
                    intensity_profiles[i,j,k] = channel[int(line_y[i,j]), int(line_x[i,j])]          
             
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


def background_noise(plot_mask, channels_data, proteins_present, size_mask, image_dim, ves_coordinates, threshold, path_to_output, exp_info):
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
            show_mask_both(int(ves_coordinates[0]), vesicle_box, mask_memb_fill, background, path_to_output, exp_info)
        
        elif protein_status == "only_septin":
            show_mask_only_sept(int(ves_coordinates[0]), vesicle_box, mask_memb_fill, background, path_to_output, exp_info)
            
        elif protein_status == "only_actin":
            show_mask_only_actin(int(ves_coordinates[0]), vesicle_box, mask_memb_fill, background, path_to_output, exp_info)
    
    return background


def plot_format(proteins_present):
    """
    Pre-processing for choosing a plotting format appropriate for the number of
    channels present.
    """
    if proteins_present[0] == True:
        if proteins_present[1] == True:
            protein_status = "both_proteins"
        else: 
            protein_status = "only_septin"
    else:
        if proteins_present[1] == True:
            protein_status = "only_actin"
    
    return protein_status

            
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
    plt.show()
    
    fig.savefig(path_to_output + "-Vesicle_" + str(vesicle_id) + "-Channels_and_Mask.png")

    
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
   plt.show()

   # Corrected: Removed doubled path injection
   fig.savefig(path_to_output + "-Vesicle_" + str(vesicle_id) + "-Channels_and_Mask.png")


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
   plt.show()
  
   # Corrected: Removed doubled path injection
   fig.savefig(path_to_output + "-Vesicle_" + str(vesicle_id) + "-Channels_and_Mask.png")


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
    """
    comment = []
    death_mark = False
    
    # Find peaks with adjusted parameters for better detection
    peaks, properties = find_peaks(radial_profile_memb, 
                                   height=np.max(radial_profile_memb) * 0.1,  # At least 10% of max
                                   distance=5, 
                                   prominence=np.max(radial_profile_memb) * 0.05)  # At least 5% prominence
    
    if not np.any(peaks):
        index_border_in  = 0
        index_border_out = 0 
        comment          = ["no_memb_peak"]
        death_mark       = True
        return index_border_in, index_border_out, comment, death_mark
    
    # If only one peak, use it
    if len(peaks) == 1:
        chosen_peak = 0
    else:
        # Multiple peaks - choose the best one based on several criteria
        peak_scores = []
        
        for i, peak_pos in enumerate(peaks):
            score = 0
            
            # Criterion 1: Prefer peaks closer to the expected radius (higher weight)
            distance_to_expected = abs(peak_pos - radius)
            distance_score = 1.0 / (1.0 + distance_to_expected / radius)  # Normalized distance score
            score += distance_score * 3.0  # High weight for position
            
            # Criterion 2: Prefer higher peaks
            height_score = radial_profile_memb[peak_pos] / np.max(radial_profile_memb)
            score += height_score * 2.0  # Medium weight for height
            
            # Criterion 3: Prefer peaks with good prominence
            prominence_score = properties['prominences'][i] / np.max(properties['prominences'])
            score += prominence_score * 1.0  # Lower weight for prominence
            
            # Criterion 4: Penalize peaks that are too early (likely noise)
            if peak_pos < radius * 0.3:  # If peak is less than 30% of expected radius
                score *= 0.5  # Reduce score significantly
            
            # Criterion 5: Penalize peaks that are too late (likely artifacts)
            if peak_pos > radius * 2.0:  # If peak is more than 2x expected radius
                score *= 0.3  # Reduce score significantly
                
            peak_scores.append(score)
        
        # Choose the peak with the highest score
        chosen_peak = np.argmax(peak_scores)
        
        # Add comment if we had to choose among many peaks
        if len(peaks) > 2:
            comment.append("multiple_peaks")
    
    # Calculate width at half maximum for the chosen peak
    try:
        width_half_max = peak_widths(radial_profile_memb, [peaks[chosen_peak]], rel_height=0.5)
        index_border_in = int(np.rint(width_half_max[2][0]))
        index_border_out = int(np.rint(width_half_max[3][0]))
    except:
        # Fallback if width calculation fails
        index_border_in = max(0, peaks[chosen_peak] - 5)
        index_border_out = min(len(radial_profile_memb) - 1, peaks[chosen_peak] + 5)
        comment.append("width_calc_failed")
    
    # Quality checks
    if index_border_out - index_border_in > radius/4:
        comment.append("wide_peak")
    
    if np.average(radial_profile_memb[:index_border_in]) >= 0.3 * radial_profile_memb[peaks[chosen_peak]]:
        comment.append("lipids_inside")
    
    if len(radial_profile_memb) > index_border_out and np.average(radial_profile_memb[index_border_out:]) >= 0.4 * radial_profile_memb[peaks[chosen_peak]]:
        comment.append("clusters_outside")
    
    return index_border_in, index_border_out, comment, death_mark

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
        if cv_membrane > 0.4:
            comments.append("very_irregular_membrane")
        elif cv_membrane > 0.3:
            comments.append("irregular_membrane")
    
    # NEW CHECK 1: Look for sharp transitions
    if len(membrane_angular) > 10:
        # Calculate differences between adjacent points
        membrane_diff = np.abs(np.diff(membrane_angular))
        max_transition = np.max(membrane_diff)
        mean_signal = np.mean(membrane_angular)
        
        # Flag if there are very sharp transitions
        if max_transition > mean_signal * 0.5:  # Transition > 50% of mean signal
            comments.append("sharp_membrane_transitions")
    
    # NEW CHECK 2: Look for plateau-like behavior (flat regions)
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
    
    # NEW CHECK 3: Check for protein clustering that might affect localization
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
    
    # NEW CHECK 4: Check for asymmetric membrane profiles
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
    """
    angular_profiles = np.ones_like(intensity_profiles[0,:,:])
    
    for i in range(num_channels):
        angular_profiles[:,i] = np.average(intensity_profiles[pixels_to_remove+index_border_in:pixels_to_remove+index_border_out,:,i], axis=0)
    
    return angular_profiles


def localization(num_channels, angular_profiles, radial_profiles, index_border_in, index_border_out, radius, size_central_area, pixels_to_remove):
    """
    Quantification of the protein localization on the membrane.
    """
    comment = []
    
    num_proteins   = num_channels - 1 
    index_centre   = int(size_central_area * (radius-pixels_to_remove)) 
 
    loc_numerator  = np.zeros(num_proteins)
    loc_denominator = np.zeros(num_proteins)
    localization   = np.zeros(num_proteins) 

    for i in range(num_proteins):
        ## Calculation or relative standard deviation of angular protein signal:
        rsd = np.std(angular_profiles[:,i+1])/np.mean(angular_profiles[:,i+1])
        
        if rsd > 0.8:
            comment = ["high_rsd"]
            
        ## Median value of protein channel:
        median  = np.median(angular_profiles[:,i+1])
        centre  = np.average(radial_profiles[0:index_centre,i+1]) 
        
        ## Localization definition:
            
        loc_numerator[i]   = median - centre
        loc_denominator[i] = median
        
        localization[i]   = np.clip(loc_numerator[i]/loc_denominator[i],0,None)
    
    return localization, comment


def plot_intensity_profiles(plot_intensity_profiles, proteins_present, parameters_sizes, channels_data, ves_coordinates, image_dim, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, background, localization, path_to_output, exp_info):
    """
    Optional plotting of the calculated intensity radial and angular profiles.
    """
    if plot_intensity_profiles == True:
        
        num_channels     = len(channels_data[0,0,:])
        size_box = int(parameters_sizes[1] * ves_coordinates[3])

        image_box = np.zeros((2*size_box, 2*size_box, num_channels))
    
        for i in range(num_channels):
            image_box[:,:,i] = zoom_in_vesicle(channels_data[:,:,i], parameters_sizes[1], image_dim, ves_coordinates)
    
        protein_status = plot_format(proteins_present)    

        if protein_status == "both_proteins":
            plot_int_prof_both(int(ves_coordinates[0]), ves_coordinates[3], parameters_sizes[2], along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info)
    
        elif protein_status == "only_septin":
            plot_int_prof_only_septin(int(ves_coordinates[0]), ves_coordinates[3], parameters_sizes[2], along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info)
        
        elif protein_status == "only_actin":
            plot_int_prof_only_actin(int(ves_coordinates[0]), ves_coordinates[3], parameters_sizes[2], along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info)


def plot_int_prof_both(vesicle_id, radius, size_central_area, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info):
    """
    Plotting of the intensity radial and angular profiles (Both proteins).
    """
    
    ## Intensity radial profile:
        
    fig, axes = plt.subplot_mosaic("AB;AC;AD;AE", width_ratios = [3, 1], dpi=125)
    axes["A"].set_title('Intensity radial profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(along_radius, radial_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(along_radius, radial_profiles[:,1], c = 'magenta', label = 'septin')
    axes["A"].plot(along_radius, radial_profiles[:,2], c = 'gold', label = 'actin')
    axes["A"].axvline(x = along_radius[index_border_in], c = 'red', label = 'border in', ls="--")
    axes["A"].axvline(x = along_radius[index_border_out], c = 'orange', label = 'border out', ls="--")
    axes["A"].axvline(x = radius, c = 'black', label = 'detected radius', ls=":")
    axes["A"].axvline(x = size_central_area*radius, c ='green', label = 'central area border')
    axes["A"].set(xlabel = 'radius (px)', ylabel = 'I (a.u.)')
    axes["A"].legend(loc="upper left", fontsize=8)
    axes["B"].set_title("Membrane channel", fontsize = 10, pad =-1)
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan")
    axes["B"].set_axis_off()
    axes["C"].set_title("Septin channel", fontsize = 10, pad =-1)
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced")
    axes["C"].set_axis_off()
    axes["D"].set_title("Actin channel", fontsize = 10, pad =-1)
    axes["D"].imshow(image_box[:,:,2], cmap = "cmap_yellow")
    axes["D"].set_axis_off()
    axes["E"].text(-0.2,0.7,"Background M:  " + str(round(background[0,0],4)), size = 10)
    axes["E"].text(-0.2,0.5,"Background S:   " + str(round(background[0,1],4)), size = 10)
    axes["E"].text(-0.2,0.3,"Background A:   " + str(round(background[0,2],4)), size = 10)
    axes["E"].text(-0.2,0.0,"Localization S :  " + str(round(localization[0],3)), size = 10)
    axes["E"].text(-0.2,-0.2,"Localization A :  " + str(round(localization[1],3)), size = 10)
    axes["E"].set_axis_off()
    
    fig.savefig(path_to_output + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Radial.png")
    
    ## Intensity angular profile:

    fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1], dpi=125)
    axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,1], c = 'magenta', label = 'septin')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,2], c = 'gold', label = 'actin')
    axes["A"].legend(loc="upper right")
    axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'I (a.u.)')
    axes["B"].set_title("Membrane channel", fontsize = 10, pad =-1)
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan")
    axes["B"].set_axis_off()
    axes["C"].set_title("Septin channel", fontsize = 10, pad =-1)
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced")
    axes["C"].set_axis_off()
    axes["D"].set_title("Actin channel", fontsize = 10, pad =-1)
    axes["D"].imshow(image_box[:,:,2], cmap = "cmap_yellow")
    axes["D"].set_axis_off()
    
    fig.savefig(path_to_output + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Angular.png")

    
def plot_int_prof_only_septin(vesicle_id, radius, size_central_area, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info):
    """
    Plotting of the intensity radial and angular profiles (Septin only).
    """
    ## Intensity radial profile:
        
    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [6, 2], dpi=125)
    axes["A"].set_title('Intensity radial profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(along_radius[2:], radial_profiles[2:,0]/np.max(radial_profiles[2:,0]), c = 'cyan', label = 'Membrane channel')
    axes["A"].plot(along_radius[2:], radial_profiles[2:,1]/np.max(radial_profiles[2:,1]), c = 'magenta', label = 'Septin channel')
    axes["A"].set(xlabel = 'radius (px)', ylabel = 'normalized intensity (a.u.)')
    axes["B"].set_title("Membrane channel", fontsize = 10, pad =-1)
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan")
    axes["B"].set_axis_off()
    axes["C"].set_title("Septin channel", fontsize = 10, pad =-1)
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced")
    axes["C"].set_axis_off()
    
    # Corrected: Removed doubled path injection
    fig.savefig(path_to_output + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Radial.png")
    
    ## Intensity angular profile:

    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [6, 2], dpi=125)
    axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,0]/max(angular_profiles[:,0]), c = 'cyan', label = 'Membrane channel')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,1]/max(angular_profiles[:,1]), c = 'magenta', label = 'Septin channel')
    axes["A"].legend(loc="upper right", fontsize=10)
    axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'normalized intensity (a.u.)')
    axes["B"].set_title("Membrane channel", fontsize = 10, pad =-1)
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan")
    axes["B"].set_axis_off()
    axes["C"].set_title("Septin channel", fontsize = 10, pad =-1)
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced")
    axes["C"].set_axis_off()
    
    # Corrected: Removed doubled path injection
    fig.savefig(path_to_output + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Angular.png")
    
    
def plot_int_prof_only_actin(vesicle_id, radius, size_central_area, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info):
    """
    Plotting of the intensity radial and angular profiles (Actin only).
    """
    ## Intensity radial profile:
        
    fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1], dpi=125)
    axes["A"].set_title('Intensity radial profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(along_radius, radial_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(along_radius, radial_profiles[:,1], c = 'gold', label = 'actin')
    axes["A"].axvline(x = along_radius[index_border_in], c = 'red', label = 'border in', ls="--")
    axes["A"].axvline(x = along_radius[index_border_out], c = 'orange', label = 'border out', ls="--")
    axes["A"].axvline(x = radius, c = 'black', label = 'detected radius', ls=":")
    axes["A"].axvline(x = size_central_area*radius, c ='green', label = 'central area border')
    axes["A"].set(xlabel = 'radius (px)', ylabel = 'I (a.u.)')
    axes["A"].legend(loc="upper left", fontsize=8)
    axes["B"].set_title("Membrane channel", fontsize = 10, pad =-1)
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan")
    axes["B"].set_axis_off()
    axes["C"].set_title("Actin channel", fontsize = 10, pad =-1)
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_yellow")
    axes["C"].set_axis_off()
    axes["D"].text(-0.2,0.7,"Background M:  " + str(round(background[0,0],4)), size = 10)
    axes["D"].text(-0.2,0.5,"Background A:   " + str(round(background[0,1],4)), size = 10)
    axes["D"].text(-0.2,0.0,"Localization A :  " + str(round(localization[0],3)), size = 10)
    axes["D"].set_axis_off()
    
    # Corrected: Removed doubled path injection
    fig.savefig(path_to_output + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Radial.png")
    
    ## Intensity angular profile:
        
    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [3, 1], dpi=125)
    axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,1], c = 'gold', label = 'actin')
    axes["A"].legend(loc="upper left")
    axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'I (a.u.)')
    axes["B"].set_title("Membrane channel", fontsize = 10, pad =-1)
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan")
    axes["B"].set_axis_off()
    axes["C"].set_title("Actin channel", fontsize = 10, pad =-1)
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_yellow")
    axes["C"].set_axis_off()

    # Corrected: Removed doubled path injection
    fig.savefig(path_to_output + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Angular.png")


def edit_output(path_to_output, exp_info, ves_coordinates, background, localization, comment):
    """
    Adds all relevant information for a single vesicle in a row of the output file.
    """
    list_exp_info        = list(exp_info[:-1])
    list_ves_coordinates = list(ves_coordinates)
    list_background      = list(background[0,:])
    list_localization    = list(localization)


    vesicle_row = list_exp_info +  list_ves_coordinates + list_background + list_localization + comment 
    
    # Corrected: Removed doubled path injection
    with open(path_to_output + "-Output.csv", "a", newline='') as output_file:
               
        writer = csv.writer(output_file)
        writer.writerow(vesicle_row)