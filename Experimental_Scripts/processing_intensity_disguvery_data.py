# -*- coding: utf-8 -*-
"""
This code reads the intensity profiles calculated by DisGUVery allowing for 
possible further processing. 

The current version doesn't fully accommodate an actin channel.
Further modifications are needed for that.

In the current version the text image is also read for convenience in the 
interpertation of results and the intensity profiles are simply printed.
 
For convenience certain parts of the code should be turned into functions at a 
later stage of development. 

Input
-------------------------------------------------------------------------------
- The path to the text images of the channel_radials should always be given by the user.
- The path to the detected_vesicles output csv file of DisGUVery should always 
  be given by the user. 
  NOTE: The '#' at the beginning of the first column's title in the csv file must
        be manually deleted by the user before running the code.
- The path to the intesnity_profiles output csv files of DisGUVery should always 
  be given by the user.
  NOTE: The '# #' at the beginning of the first column's title in the csv fil, as
        well as the second empty row starting with a '#' must be manually deleted
        by the user before running the code.

Short description
-------------------------------------------------------------------------------
1. The text images and the csv are read. The csv columns are turned into separate
   arrays for convenience. 
2. All parameter values are defined by the user.
3. The indices of the channel changes are determined. Used to automatically 
   "segment" the inputed data for the different vesicles. Process takes place 
   separately for the radial and angular profile data as the number of elements
   in the columns differ. 
4. Plotting intensity profiles:
   The intensity profiles are printed along with a zoomed in image of the vesicle
   in all channels for convenience of data interpatation. 

Parameters
-------------------------------------------------------------------------------
length_excess : float
       The factor multiplying the radius of the vesicles to define the area 
       depicted in the zoomed in picture.

Output
-------------------------------------------------------------------------------
The current version plots the radial and angular intensity profiles calculated
by DisGUVery along with the corresponding identified vesicle's zoomed in image.  

NOTE: The angle 0 dorresponds to a different direction compared to intensity_profiles.py
      Should adjust that for easier comparison. 
-------------------------------------------------------------------------------  
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                                                            # to access csv file 
from matplotlib.colors import LinearSegmentedColormap                          # to create colormaps matching our conventions 

### 1. READING FILES ----------------------------------------------------------

## Reading channels from text image files:

txt_membrane = np.loadtxt('Composite_membrane_text.txt')
txt_septin   = np.loadtxt('Composite_septin_text.txt')

## Reading csv file of detected vesicles: 
    
detection_ch = pd.read_csv(r'Composite_detected_vesicles.csv')
xc           = pd.DataFrame(detection_ch, columns= [' xc (pix)']).to_numpy()
yc           = pd.DataFrame(detection_ch, columns= [' yc (pix)']).to_numpy()
radius       = pd.DataFrame(detection_ch, columns= [' radius (pix)']).to_numpy()

## Reading radial intesnity file: 

intensity_radial_file = pd.read_csv(r'Composite_results_m_roi_center_s_roi_corner_radial_profiles.csv')
identity_radial       = pd.DataFrame(intensity_radial_file, columns= [' Ves ID']).to_numpy()  
channel_radial        = pd.DataFrame(intensity_radial_file, columns= [' Channel']).to_numpy()
radius_radial         = pd.DataFrame(intensity_radial_file, columns= [' Mean radius (pix)']).to_numpy()
intensity_radial      = pd.DataFrame(intensity_radial_file, columns= [' Mean Intensity (a.u.)']).to_numpy()

## Reading angular intesnity file: 

intensity_angular_file = pd.read_csv(r'Composite_results_m_roi_center_s_roi_corner_angular_profiles.csv')
identity_angular              = pd.DataFrame(intensity_angular_file, columns= [' Ves ID']).to_numpy()  
channel_angular               = pd.DataFrame(intensity_angular_file, columns= [' Channel']).to_numpy()
radius_angular                = pd.DataFrame(intensity_angular_file, columns= [' Mean theta (deg)']).to_numpy()
intensity_angular             = pd.DataFrame(intensity_angular_file, columns= [' Mean Intensity (a.u.)']).to_numpy()

### 2. DEFINING PARAMETER VALUES ---------------------------------------------- 

length_excess = 1.3                                                            # the factor multiplying the radi of the vesicles to determine the area of the zoomed in picture

### 3. FINDING THE INDICES OF CHANNEL CHANGES ---------------------------------

## Determining total number of vesicles and channels, as well as useful dimensions from the radial file:

num_vesicles_radial = int(identity_radial[-1])    
num_channels_radial = int(channel_radial[-1])

full_size_radial    = len(identity_radial)
num_segments_radial = num_vesicles_radial*num_channels_radial

## Finding where the different channels start in the radial file:
    
split_indices_radial = np.array(0)

for i in range(full_size_radial-1):
    if channel_radial[i+1] != channel_radial[i]:
        split_indices_radial = np.append(split_indices_radial, i+1)
split_indices_radial = np.append(split_indices_radial, full_size_radial)

## Determining total number of vesicles and channels, as well as useful dimensions from the angular file:

num_vesicles_angular = int(identity_angular[-1])    
num_channels_angular = int(channel_angular[-1])

full_size_angular    = len(identity_angular)
num_segments_angular = num_vesicles_angular*num_channels_angular

## Finding where the different channels start in the angular file:
    
split_indices_angular = np.array(0)

for i in range(full_size_angular-1):
    if channel_angular[i+1] != channel_angular[i]:
        split_indices_angular = np.append(split_indices_angular, i+1)
split_indices_angular = np.append(split_indices_angular, full_size_angular)

### 4. PLOTTING INTENSITY PROFILES --------------------------------------------

## Defining colormaps that match the LUTs of ImageJ used conventionally:
    
colors          = ["black","cyan"]
nodes           = [0.0, 1.0]
my_cmap_cyan    = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

colors          = ["black","magenta"]
nodes           = [0.0, 1.0]
my_cmap_magenta = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    
for j in range(0, num_segments_radial,2): 
    
    ## Creating zoomed-in picture of vesicle for conveninece:
    
    vesicle_id = int(j/2+1) - 1 
    zoom_in_vesicle_membrane = txt_membrane[int(yc[vesicle_id]-length_excess*radius[vesicle_id]):int(yc[vesicle_id]+length_excess*radius[vesicle_id]), int(xc[vesicle_id]-length_excess*radius[vesicle_id]):int(xc[vesicle_id]+length_excess*radius[vesicle_id])]
    zoom_in_vesicle_septin   = txt_septin[int(yc[vesicle_id]-length_excess*radius[vesicle_id]):int(yc[vesicle_id]+length_excess*radius[vesicle_id]), int(xc[vesicle_id]-length_excess*radius[vesicle_id]):int(xc[vesicle_id]+length_excess*radius[vesicle_id])]
   
    ## Ploting vesicle's radial profile:
        
    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [3, 1])
    axes["A"].set_title('Intensity radial profile of Vesicle ' + str(int(j/2+1)))
    axes["A"].plot(radius_radial[split_indices_radial[j]:split_indices_radial[j+1]], intensity_radial[split_indices_radial[j]:split_indices_radial[j+1]], c = 'cyan', label = 'membrane')
    if num_channels_radial >= 2 :
        axes["A"].plot(radius_radial[split_indices_radial[j+1]:split_indices_radial[j+2]], intensity_radial[split_indices_radial[j+1]:split_indices_radial[j+2]], c = 'magenta', label = 'septin')
    if num_channels_radial == 3 :
        axes["A"].plot(radius_radial[split_indices_radial[j+2]:split_indices_radial[j+3]], intensity_radial[split_indices_radial[j+2]:split_indices_radial[j+3]], c = 'yellow', label = 'actin')
    axes["A"].set(xlabel = 'radius (px)', ylabel = 'I (a.u.)')
    axes["A"].legend(loc="upper left")
    axes["B"].set_title("Membrane channel")
    axes["B"].imshow(zoom_in_vesicle_membrane, cmap = my_cmap_cyan)
    axes["B"].set_axis_off()
    if num_channels_radial >= 2 :
        axes["C"].set_title("Septin channel")
        axes["C"].imshow(zoom_in_vesicle_septin, cmap = my_cmap_magenta)
        axes["C"].set_axis_off()    

    ## Ploting vesicle's angular profile:

    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [3, 1])
    axes["A"].set_title('Intensity angular profile of Vesicle ' + str(int(j/2+1)))
    axes["A"].plot(radius_angular[split_indices_angular[j]:split_indices_angular[j+1]], intensity_angular[split_indices_angular[j]:split_indices_angular[j+1]], c = 'cyan', label = 'membrane')
    if num_channels_angular >= 2 :
        axes["A"].plot(radius_angular[split_indices_angular[j+1]:split_indices_angular[j+2]], intensity_angular[split_indices_angular[j+1]:split_indices_angular[j+2]], c = 'magenta', label = 'septin')
    if num_channels_angular == 3 :
        axes["A"].plot(radius_angular[split_indices_angular[j+2]:split_indices_angular[j+3]], intensity_angular[split_indices_angular[j+2]:split_indices_angular[j+3]], c = 'yellow', label = 'actin')
    axes["A"].set(xlabel = 'radius (px)', ylabel = 'I (a.u.)')
    axes["A"].legend(loc="upper left")
    axes["B"].set_title("Membrane channel")
    axes["B"].imshow(zoom_in_vesicle_membrane, cmap = my_cmap_cyan)
    axes["B"].set_axis_off()
    if num_channels_angular >= 2 :
        axes["C"].set_title("Septin channel")
        axes["C"].imshow(zoom_in_vesicle_septin, cmap = my_cmap_magenta)
        axes["C"].set_axis_off() 