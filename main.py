# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:33:44 2022

@author: kkourkoulou (Updated by Gemini)

Script to calculate the radial and angular intensity profiles of GUVs 
encapsulating septin and/or actin and to quantify the corresponding protein
localization on the vesicle membrane. 
 
main.py     :    main file to run the script
skeleton.py :    file with all necessary functions in main.py
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

#------------------------- INPUT --------------------------------

## Specify paths to the directories containing the data:
path_membrane     = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\251128_LinearCortex\ImageSequences\C1"
path_detected     = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\251128_LinearCortex\ImageSequences\C1\Detected"
path_septin       = r"" 
path_actin        = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\251128_LinearCortex\ImageSequences\C3"

# Define where you want the output saved 
path_to_output_root = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\Output"

## Defining which proteins are present (True) or absent (False):
Septin            = False       
Actin             = True

## Defining parameter values for the analysis:
num_angles        = 360            # number of different equally-spaced linear profiles to be taken into account for a single vesicle
length_excess     = 1.2            # the length (unit of measure: each vesicles' radius) of the linear profiles to be considered
dr                = 1              # the step (in px) between the points sampled along the radial direction for the linear profiles calculation  
size_mask         = 3              # the length (unit of measure: each vesicles' radius) of the side of the square zoomed-in image, centered at each vesicle, to be considered during background signal calculation
size_view         = length_excess  # the length (unit of measure: each vesicles' radius) of the side of the square zoomed-in image, centered at each vesicle, used for visual inspection along the final results
size_central_area = 1/4            # the central area radius (unit of measure: each vesicles' radius) considered during localization quantification

## Deciding on optional output:
plot_detected_centres   = True     # for inspection of the whole region and the positions of the detected vesicles
threshold_method_manual = False    # for manual selection of the threshold method for the background signal calculation (default: Li threshold)  
plot_mask               = False    # for visual inspection of the calculated masks for the background signal calculation
plot_int_profiles       = True     # for plotting the radial and angular intensity profiles

## PARALLEL PROCESSING SETTINGS
n_jobs = -1  # -1 means use all available CPUs. Set to 1 for standard sequential processing.

#----------------------------------------------------------------

#------------------------ PREPARATION ---------------------------
## Rearranging input:
proteins_present    = np.array((Septin, Actin), dtype = bool)

# NOTE: skeleton.py expects these as Arrays to use index access (e.g. parameters[0])
parameters_profiles = np.array((num_angles, length_excess, dr))
parameters_sizes    = np.array((size_mask, size_view, size_central_area))

## Finding experiment and image information by scanning the Membrane (C1) folder:
exp_info_all_sets   = skl.find_projects_info(path_membrane)

# Handle case where only 1 set is found (ensure 2D array)
if exp_info_all_sets.ndim == 1:
    num_sets = 1
    exp_info_all_sets = exp_info_all_sets.reshape(1, -1)
else:
    num_sets = len(exp_info_all_sets[:,0])

for s in range(num_sets):
    
    exp_info = exp_info_all_sets[s,:]
    print(f"--- Processing Set {s+1}/{num_sets}: {exp_info[2]} ---")
    
    ## Create Directory Structure: Output/Date_Exp/Region_ID
    final_output_path = skl.create_directory_structure(path_to_output_root, exp_info)
    
    ## Create CSV in the new specific folder
    output_csv_path = skl.create_output_file(final_output_path, proteins_present)

    ## Reading image data:
    # UPDATED: Now unpacks 3 values (data, coords, pixel_size)
    channels_data, coordinates, pixel_size = skl.read_files(path_membrane, path_septin, path_actin, path_detected, exp_info, proteins_present)
    
    ## Creating colormaps
    skl.color_maps()

    ## Deriving useful parameters: 
    num_vesicles = len(coordinates[:,0])
    # Ensure dimensions are read correctly (Y, X)
    image_dim  = np.array((channels_data.shape[1], channels_data.shape[0])) 

    ## Optional plotting of the membrane channel with annotated vesicle centres:
    skl.detected_centres(plot_detected_centres, num_vesicles, channels_data[:,:,0], coordinates[:,1], coordinates[:,2], final_output_path, exp_info)

    ## Optional manual choice of thresholding method (default method:Li):
    # Pass first vesicle coordinates for threshold estimation
    threshold_membrane = skl.define_threshold(threshold_method_manual, channels_data[:,:,0], size_mask, image_dim, coordinates[0,:]) 
    
    # List to collect radii for distribution plot
    all_refined_radii = []

#----------------------------------------------------------------

#--------------------------- MAIN (PARALLEL) --------------------
    
    print(f"Starting parallel processing for {num_vesicles} vesicles (n_jobs={n_jobs})...")

    # Run processing in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(skl.process_single_vesicle)(
            coordinates[i,:], channels_data, image_dim, parameters_profiles, 
            proteins_present, size_mask, threshold_membrane, final_output_path, 
            exp_info, parameters_sizes, plot_int_profiles, plot_mask, pixel_size
        ) for i in range(num_vesicles)
    )

    # Collect results and write to CSV sequentially
    print("Saving results...")
    
    # We open in 'append' mode ('a') because create_output_file already wrote the header
    with open(output_csv_path, "a", newline='') as output_file:
        writer = csv.writer(output_file)
        
        for row, refined_radius in results:
            writer.writerow(row)
            if refined_radius is not None and refined_radius > 0:
                all_refined_radii.append(refined_radius)

    # Plot Size Distribution
    skl.plot_size_distribution(all_refined_radii, final_output_path)
    print("Done with set!")

print("All sets processed.")