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

import numpy as np
import skeleton as skl

#------------------------- INPUT --------------------------------

## Specify paths to the directories containing the data:
# We use raw strings (r"...") to ensure Windows backslashes are read correctly
path_membrane     = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\251110_BranchedCortex\ImageSequences\C1"
path_detected     = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\251110_BranchedCortex\ImageSequences\C1\Detected"
# Septin path is left empty for now as Septin = False, but kept for future use
path_septin       = r"" 
path_actin        = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\251110_BranchedCortex\ImageSequences\C3"

# Define where you want the output saved 
path_to_output_root = r"M:\tnw\bn\gk\NN\2_Data-Analysis\Protein_Localization\251110_BranchedCortex\ImageSequences"

## Defining which proteins are present (True) or absent (False):
Septin            = False       
Actin             = True

## Defining parameter values for the analysis:
num_angles        = 360            # number of different equally-spaced linear profiles to be taken into account for a single vesicle
length_excess     = 1.5            # the length (unit of measure: each vesicles' radius) of the linear profiles to be considered
dr                = 1              # the step (in px) between the points sampled along the radial direction for the linear profiles calculation  
size_mask         = 3              # the length (unit of measure: each vesicles' radius) of the side of the square zoomed-in image, centered at each vesicle, to be considered during background signal calculation
size_view         = length_excess  # the length (unit of measure: each vesicles' radius) of the side of the square zoomed-in image, centered at each vesicle, used for visual inspection along the final results
size_central_area = 1/4            # the central area radius (unit of measure: each vesicles' radius) considered during localization quantification

## Deciding on optional output:
plot_detected_centres   = True     # for inspection of the whole region and the positions of the detected vesicles
threshold_method_manual = False    # for manual selection of the threshold method for the background signal calculation (default: Li threshold)  
plot_mask               = False    # for visual inspection of the calculated masks for the background signal calculation
plot_int_profiles       = True     # for plotting the radial and angular intensity profiles

#----------------------------------------------------------------

#------------------------ PREPARATION ---------------------------
## Rearranging input:
proteins_present    = np.array((Septin, Actin), dtype = bool)
parameters_profiles = np.array((num_angles, length_excess, dr))
parameters_sizes    = np.array((size_mask, size_view, size_central_area))

## Finding experiment and image information by scanning the Membrane (C1) folder:
exp_info_all_sets   = skl.find_projects_info(path_membrane)

if exp_info_all_sets.ndim == 1:
    num_sets = 1
else:
    num_sets = len(exp_info_all_sets[:,0])

for s in range(num_sets):
    
    if exp_info_all_sets.ndim == 1:
        exp_info = exp_info_all_sets
    else:
        exp_info = exp_info_all_sets[s,:]
    

    ## Creating output file:
    path_to_output = path_to_output_root + "\\" + exp_info[3]
    skl.create_output(path_to_output, exp_info, proteins_present)

    ## Reading image data (Passing all specific directories):
    channels_data, coordinates = skl.read_files(path_membrane, path_septin, path_actin, path_detected, exp_info, proteins_present)

    ## Creating colormaps following the chosen conventions (if colormaps are not already registered in matplotlib):
    skl.color_maps()

    ## Deriving useful parameters: 
    num_vesicles = len(coordinates[:,0])
    num_channels = len(channels_data[0,0,:])
    image_dim  = np.array((len(channels_data[0,:,0]), len(channels_data[:,0,0])))

    ## Optional plotting of the membrane channel with annotated vesicle centres:
    skl.detected_centres(plot_detected_centres, num_vesicles, channels_data[:,:,0], coordinates[:,1], coordinates[:,2], path_to_output, exp_info)

    ## Optional manual choice of thresholding method (default method:Li):
    threshold_membrane = skl.define_threshold(threshold_method_manual, channels_data[:,:,0], size_mask, image_dim, coordinates[0,:]) 


#----------------------------------------------------------------

#--------------------------- MAIN -------------------------------

    for i in range(num_vesicles):
        ## Initializing list for tracking triggered conditions for automatic vesicle rejection:
        comment = []

        ## Calculating the individual linear profiles to be considered:
        intensity_profiles, along_radius, theta, death_mark = skl.linear_profiles(channels_data, coordinates[i,:], image_dim, parameters_profiles)
        ## Output for vesicle profiles extending out of the image margins is set to 0:
        if death_mark == True:
            background = np.zeros((1, num_channels))
            localization = np.zeros(num_channels-1)
            comment = ["margins"]
            skl.edit_output(path_to_output, exp_info, coordinates[i,:], background, localization, comment)
            continue
        
        ## Calculating the background signal based on the vesicles' vicinity (with optional plotting of the corresponding calculated mask):
        background = skl.background_noise(plot_mask, channels_data, proteins_present, size_mask, image_dim, coordinates[i,:], threshold_membrane, path_to_output, exp_info)

        ## Background signal removal:
        intensity_profiles_corrected = skl.background_correction(num_channels, intensity_profiles, background)

        ## Calculation of the collective radial profile of the requested vesicle:
        radial_profiles = skl.radial_profile(num_channels, intensity_profiles_corrected)
     
        ## Correction for bias of the first pixels due to multi-counting:
        pixels_to_remove = 2
        radial_profiles  = radial_profiles[pixels_to_remove:,:]
        along_radius     = along_radius[pixels_to_remove:]
        
        ## Membrane border detection:
        index_border_in, index_border_out, comment_peak, death_mark_peak = skl.membrane_detection(radial_profiles[:,0], coordinates[i,3])
        ## Registration of the triggered conditions for automatic vesicle rejection:
        if comment_peak:
            comment = comment + comment_peak
        ## Output for profiles where no peak was detected is set to 0:
        if death_mark_peak == True:
            background = np.zeros((1, num_channels))
            localization = np.zeros(num_channels-1)
            comment = comment_peak
            skl.edit_output(path_to_output, exp_info, coordinates[i,:], background, localization, comment)
            continue
        
        ## Marking of vesicles with low septin signal inside:
        for j in range(num_channels-1):
            if max(radial_profiles[:index_border_out,j+1]) <= 5:
                comment = comment + ["low_protein"]
        
        ## Calculation of the angular profile of the vesicle under study:
        angular_profiles = skl.angular_profile(num_channels, intensity_profiles_corrected, index_border_in, index_border_out, pixels_to_remove)

        ## Quality check for membrane uniformity:
        quality_comments = skl.check_membrane_quality(angular_profiles, radial_profiles, coordinates[i,3])
        if quality_comments:
            comment = comment + quality_comments
            print(f"Vesicle {i+1}: Quality flags - {quality_comments}")
        
        ## Output for profiles where two borders coincided is set to 0:
        if index_border_in == index_border_out :
            comment = comment + ["no_memb_detected"]
            comment = [', '.join(comment)]
            background = np.zeros((1, num_channels))
            localization = np.zeros(num_channels-1)
            skl.edit_output(path_to_output, exp_info, coordinates[i,:], background, localization, comment)
            continue
        
        ## Quantification of protein localization on the membrane:
        localization, comment_loc = skl.localization(num_channels, angular_profiles, radial_profiles, index_border_in, index_border_out, coordinates[i,3], size_central_area, pixels_to_remove)
      
        ## Registration of the triggered conditions for automatic vesicle rejection:
        if np.isnan(localization) == True:
            comment = comment + ["no_memb_detected"]
        elif np.isinf(localization) == True:
            comment = comment + ["zero_at_centre"]
        if comment_loc:
            comment = comment + comment_loc
            
      
        ## Optional plotting the intensity profiles:
        skl.plot_intensity_profiles(plot_int_profiles, proteins_present, parameters_sizes, channels_data, coordinates[i,:], image_dim, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, background, localization, path_to_output, exp_info)
    
        ## Saving data in output file:
        if not comment:
            comment = ["OK"]
        else:
            comment = [', '.join(comment)]
            
        skl.edit_output(path_to_output, exp_info, coordinates[i,:], background, localization, comment)