# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:33:44 2022

@author: kkourkoulou


Script to calculate and analyze the intensity profiles of GUVs detected with 
DisGUVery. 

Main file to run the script.

"""

import numpy as np
import skeleton as skl
import matplotlib.pyplot as plt


#------------------------- INPUT --------------------------------

## Specify path to the data directory and experiment information:
path_to_script    = "C:\\elise-katerina_mep-main"

## Defining which proteins are present (True) or absent (False):
Septin            = False       
Actin             = True

## Defining parameter values for the analysis:
num_angles        = 360            # number of different equally-spaced linear profiles to be taken into account for a single vesicle
length_excess     = 1.5            # the length (unit of measure: each vesicles' radius) of the linear profiles to be considered
dr                = 1              # the step (in px) between the points sampled along the radial direction for the linear profiles calculation  
size_mask         = 3              # the length (unit of measure: each vesicles' radius) of the side of the square zoomed-in image, centered at each vesicle, to be considered during background noise calculation
size_view         = length_excess  # the length (unit of measure: each vesicles' radius) of the side of the square zoomed-in image, centered at each vesicle, used for visual inspection along the final results
size_central_area = 1/4            # the central area radius (unit of measure: each vesicles' radius) considered during localization quantification

## Deciding on optional output:
plot_detected_centres   = True
threshold_method_manual = False 
plot_mask               = False 
plot_int_profiles       = True 

#----------------------------------------------------------------

#------------------------ PREPARATION ---------------------------
## Rearranging input:
proteins_present    = np.array((Septin, Actin), dtype = bool)
parameters_profiles = np.array((num_angles, length_excess, dr))
parameters_sizes    = np.array((size_mask, size_view, size_central_area))

## Finding experiment and image information for all data sets in "data" folder:
exp_info_all_sets   = skl.find_projects_info(path_to_script)

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
    path_to_output    = path_to_script + "\\output\\" + exp_info[3]
    skl.create_output(path_to_output, exp_info, proteins_present)

    ## Reading image data:
    channels_data, coordinates = skl.read_files(path_to_script, exp_info, proteins_present)

    ## Creating colormaps following the chosen conventions (if colormaps are not already registered in matplotlib):
    colormaps = skl.color_maps()
    
    ## Deriving useful parameters: SOME ADJUSTMENTS ARE PROBABLY NEEDED HERE AT THE END
    num_vesicles = len(coordinates[:,0])
    
    # Check if channels_data is 2D or 3D:
    if len(channels_data.shape) == 2:
        # Convert 2D array to 3D array with one channel
        channels_data = channels_data.reshape(channels_data.shape[0], channels_data.shape[1], 1)
        print("Converted 2D channels_data to 3D with shape:", channels_data.shape)

    # Now we can safely access the third dimension
    num_channels = channels_data.shape[2]
    image_dim = np.array((channels_data.shape[1], channels_data.shape[0]))
    
    # Display channels
    plt.figure(figsize=(8, 6))
    
    # Display membrane channel (index 0)
    plt.subplot(1, num_channels, 1)
    plt.imshow(channels_data[:,:,0], cmap=colormaps["cmap_cyan"])
    plt.title('Membrane Channel')
    plt.colorbar()
    
    # Display additional channels if present (like Septin)
    if num_channels > 1:
        plt.subplot(1, num_channels, 2)
        plt.imshow(channels_data[:,:,1], cmap=colormaps["cmap_magenta"])
        plt.title('Septin Channel')
        plt.colorbar()
    
    # Display more channels if present (like Actin)
    if num_channels > 2:
        plt.subplot(1, num_channels, 3)
        plt.imshow(channels_data[:,:,2], cmap=colormaps["cmap_yellow"])
        plt.title('Actin Channel')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(path_to_output + "\\all_channels.png")
    plt.close()

    ## Optional plotting of the membrane channel with annotated vesicle centres:
    skl.detected_centres(plot_detected_centres, num_vesicles, channels_data[:,:,0], coordinates[:,1], coordinates[:,2], path_to_output, exp_info)

    ## Optional manual choice of thresholding method (default method:Li):
    threshold_membrane = skl.define_threshold(threshold_method_manual, channels_data[:,:,0], size_mask, image_dim, coordinates[0,:]) 


#----------------------------------------------------------------

#--------------------------- MAIN -------------------------------

    for i in range(num_vesicles):
        comment = []

        ## Calculating the linear profiles to be considered:
        intensity_profiles, along_radius, theta, death_mark = skl.linear_profiles(channels_data, coordinates[i,:], image_dim, parameters_profiles)
        if death_mark == True:
            background = np.zeros((1, num_channels))
            localization = np.zeros(num_channels-1)
            comment = ["margins"]
            skl.edit_output(path_to_output, exp_info, coordinates[i,:], background, localization, comment)
            continue
        
        ## Calculating the background noise based on the vesicles' vicinity:
        ## Additionally, optional plotting of the masks for visual inspection is possible.
        background = skl.background_noise(plot_mask, channels_data, proteins_present, size_mask, image_dim, coordinates[i,:], threshold_membrane, path_to_output, exp_info)

        ## Background noise correction:
        intensity_profiles_corrected = skl.background_correction(num_channels, intensity_profiles, background)

        ## Calculation of the collective radial profile of the requested vesicle:
        radial_profiles = skl.radial_profile(num_channels, intensity_profiles_corrected)
     
        ## Correction of central area:
        pixels_to_remove = 2
        radial_profiles  = radial_profiles[pixels_to_remove:,:]
        along_radius     = along_radius[pixels_to_remove:]
        
        ## Membrane border detection:
        index_border_in, index_border_out, comment_peak, death_mark_peak = skl.membrane_detection(radial_profiles[:,0], coordinates[i,3])
        if comment_peak:
            comment = comment + comment_peak
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
        
        ## Calculation of the angular profile of the requested vesicle:
        angular_profiles = skl.angular_profile(num_channels, intensity_profiles_corrected, index_border_in, index_border_out, pixels_to_remove)
        
        if index_border_in == index_border_out :
            comment = comment + ["no_memb_detected"]
            comment = [', '.join(comment)]
            background = np.zeros((1, num_channels))
            localization = np.zeros(num_channels-1)
            skl.edit_output(path_to_output, exp_info, coordinates[i,:], background, localization, comment)
            continue
        
        ## Quantifying protein localization on the membrane:
        localization, comment_loc = skl.localization_median(num_channels, angular_profiles, radial_profiles, index_border_in, index_border_out, coordinates[i,3], size_central_area, pixels_to_remove)
        
        if np.isnan(localization) == True:
            comment = comment + ["no_memb_detected"]
        elif np.isinf(localization) == True:
            comment = comment + ["zero_at_centre"]
        
        if comment_loc:
            comment = comment + comment_loc
            
        ## WORK IN PROGRESS: Find and avoid clusters on the membrane:
        # skl.angular_variance_assistant(localization, angular_profiles, theta, channels_data, parameters_sizes, coordinates[i,:], image_dim)    
    
        ## Plotting the intensity profiles:
        skl.plot_intensity_profiles(plot_int_profiles, proteins_present, parameters_sizes, channels_data, coordinates[i,:], image_dim, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, background, localization, path_to_output, exp_info)
    
        ## Saving data in output file:
        if not comment:
            comment = ["OK"]
        else:
            comment = [', '.join(comment)]
            
        skl.edit_output(path_to_output, exp_info, coordinates[i,:], background, localization, comment)
        

#----------------------------------------------------------------
