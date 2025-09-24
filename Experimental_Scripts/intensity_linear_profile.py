# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:32:42 2022

@author: Katerina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


### READING FILES ----------------------------------------------------------------------------------------------------
## To open the image in text format:

txt_membrane = np.loadtxt('Region 2_RAW_ch01-fr2.txt')
txt_septin   = np.loadtxt('Region 2_RAW_ch00-fr2.txt')

## Reading csv file of detected vesicles:                                      # IMPORTANT NOTE: manually delete '#' from first column's title: '# id_vesicle' in DisGUVery output file

detection_ch = pd.read_csv(r'Composite_fr2region2_detected_vesicles.csv')
identity     = pd.DataFrame(detection_ch, columns= [' id_vesicle']).to_numpy()
xc           = pd.DataFrame(detection_ch, columns= [' xc (pix)']).to_numpy()
yc           = pd.DataFrame(detection_ch, columns= [' yc (pix)']).to_numpy()
radius       = pd.DataFrame(detection_ch, columns= [' radius (pix)']).to_numpy()

num_vesicles = len(radius)

### VISUAL INSPECTION OF DETECTION -----------------------------------------------------------------------------------

plot_centres = 'True'

if plot_centres == 'True':
    plt.figure()
    plt.axis('off')
    plt.style.use('default')    # alternative viewing: 'dark_background'
    plt.imshow(txt_membrane, cmap='binary_r')
    plt.scatter(xc, yc, color='red', s=1, marker="o")
    for i in range(num_vesicles):
        plt.annotate(i+1, (xc[i]+30, yc[i]-20), c = 'red')
    
### PROCESSING VESICLE 1 (a.k.a. index = 0) --------------------------------------------------------------------------

vesicle_id = 0
## Defining number of line profiles to be taken into account:
num_segments      = 10
num_angles        = num_segments + 1
theta             = np.linspace(0, 2*np.pi, num_angles)

## Defining radial coordinate for the intensity profile:
length_excess = 1.2
dr            = 1
along_radius  = np.arange(0, int(length_excess*radius[vesicle_id]), dr)

## Defining indices of pixels along the line profiles, NOTE: rounding to nearest integer used in decision making
line_x       = np.zeros((len(along_radius), num_angles))
line_y       = np.zeros((len(along_radius), num_angles))

for i in range(len(theta)):
    line_x[:,i] = int(xc[vesicle_id]) + np.rint(along_radius * np.cos(theta[i]))
    line_y[:,i] = int(yc[vesicle_id]) + np.rint(along_radius * np.sin(theta[i]))

## Visual inspection that the line profiles' indices are as intended:

plot_lines = 'True'

if plot_lines == 'True':
    plt.figure()
    plt.axis('off')
    plt.style.use('default')    # alternative viewing: 'dark_background'
    plt.imshow(txt_membrane, cmap='binary_r')
    plt.scatter(xc, yc, color='red', s=1, marker="o")
    for i in range(num_vesicles):
        plt.annotate(i+1, (xc[i]+30, yc[i]-20), c = 'red')
    for j in range(len(theta)):
        plt.plot(line_x[:,j],line_y[:,j])

## Assigning the intencity values of the membrane and septin:

intensity_membrane  = np.zeros((len(along_radius), num_angles))
intensity_septin    = np.zeros((len(along_radius), num_angles)) 

for i in range(len(along_radius)):
    for j in range((num_angles)):            
        intensity_membrane[i,j]  = txt_membrane[int(line_y[i,j]), int(line_x[i,j])]
        intensity_septin[i,j]  = txt_septin[int(line_y[i,j]), int(line_x[i,j])]


## Visual inspection of intensity profiles:
    
plot_intensity = 'True'

if plot_intensity == 'True':
    for j in range(len(theta)-1):
        plt.figure()
        plt.style.use('default')
        plt.title('Linear intensity profile of Vesicle ' + str(vesicle_id+1) + ', angle = ' + str(round(math.degrees(theta[j])))+ '$^\circ$')
        plt.plot(along_radius, intensity_membrane[:,j], c = 'cyan', label = 'membrane')
        plt.plot(along_radius, intensity_septin[:,j], c = 'magenta', label = 'septin')
        plt.xlabel(xlabel = 'radius (px)')
        plt.ylabel(ylabel = 'I (a.u.)')
        plt.legend()
        
## Average "over all angles" the intensity plots:
    
mean_int_membr  = np.mean(intensity_membrane, axis = 1)
mean_int_sept   = np.mean(intensity_septin, axis = 1) 

membr_max_index = np.argmax(mean_int_membr)
membr_min_index = np.argmin(mean_int_membr)
half_max = mean_int_membr[membr_max_index] / 2 

zero_cross = np.where(np.diff(np.sign( mean_int_membr-half_max)))[0]

print(zero_cross[0])
print(along_radius)
       
plt.figure()   
plt.title('Vesicle ' + str(vesicle_id+1) + ': mean intensity with step: ' + str(round(360/num_angles))+ ' degrees')
plt.plot(along_radius, mean_int_membr, c = 'cyan', label = 'membrane')
plt.plot(along_radius, mean_int_sept, c = 'magenta', label = 'septin')
plt.axvline(x = zero_cross[0])
plt.axvline(x = zero_cross[1]+1)








# ## Detecting the membrane
# from scipy.signal import chirp, find_peaks, peak_widths

# peaks, _ = find_peaks(mean_int_membr, threshold = [1])

# results_half = peak_widths(mean_int_membr, peaks, rel_height=0.5)
# print(*results_half)

# plt.plot(mean_int_membr)
# plt.plot(peaks, mean_int_membr[peaks], "x")

# plt.hlines(*results_half[1:], color="C2")

# plt.show()












    
# ### Preparing function for replacment:
# def intensity_profiles(vesicle_id, num_segments, radius, xc, yc, membrane_ch, septin_ch):
#     num_angles        = num_segments + 1
#     theta             = np.linspace(0, 2*np.pi, num_angles)
    
#     ## Defining radial coordinate for the intensity profile:
#     length_excess = 1.3
#     dr            = 1
#     along_radius  = np.arange(0, int(length_excess*radius[vesicle_id]), dr)
    
#     ## Defining indices of pixels along the line profiles, NOTE: rounding to nearest integer used in decision making
#     line_x       = np.zeros((len(along_radius), num_angles))
#     line_y       = np.zeros((len(along_radius), num_angles))
    
#     for i in range(len(theta)):
#         line_x[:,i] = int(xc[vesicle_id]) + np.rint(along_radius * np.cos(theta[i]))
#         line_y[:,i] = int(yc[vesicle_id]) + np.rint(along_radius * np.sin(theta[i]))
    
#     ## Assigning the intencity values of the membrane and septin:

#     intensity_membrane  = np.zeros((len(along_radius), num_angles))
#     intensity_septin    = np.zeros((len(along_radius), num_angles)) 

#     for i in range(len(along_radius)):
#         for j in range((num_angles)):            
#             intensity_membrane[i,j]  = membrane_ch[int(line_y[i,j]), int(line_x[i,j])]
#             intensity_septin[i,j]  = septin_ch[int(line_y[i,j]), int(line_x[i,j])]
    
#     return intensity_membrane, intensity_septin, along_radius, theta
    
# intensity_membrane, intensity_septin, along_radius, theta = intensity_profiles(2, 4, radius, xc, yc, txt_membrane, txt_septin)