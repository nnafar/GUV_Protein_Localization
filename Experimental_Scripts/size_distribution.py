# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:20:23 2023

@author: Katerina
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import seaborn as sns
import csv


## Parameters:-----------------------------------------------------------------

# Files........................................................................

Septins        = "Hexamers"    
# Septins       = "Octamers"

compositions   = ["100% DOPC","0% DOPS, 5% PIP","20% DOPS, 0% PIP","20% DOPS, 1% PIP","20% DOPS, 3% PIP","20% DOPS, 5% PIP"]

# Input:
    
data_dir       = "D:\\Personal\\TU Delft\\Thesis\\Intensity profiles\\output\\New definitions\\" + Septins + "\\"
pixel_size_dir = "D:\\Personal\\TU Delft\\Thesis\\Intensity profiles\\data\\OK data\\New_All\\" + Septins + "\\pixel_micron_ratio\\" 

# Output:
path_to_output = "D:\\Personal\\TU Delft\\Thesis\\Intensity profiles\\output\\New definitions\\" + Septins + "\\With pixel size\\"

# What to include..............................................................

Combine_files          = True # Needed only once the first time the code runs
Plot_size_distribution = True 
Size_loc_correletion   = True

if Combine_files == True:
    for j in range(len(compositions)):
        
        current_composition = compositions[j]
        
    
        os.chdir(data_dir) 
        initial_csv  = pd.read_csv("Auto_filtered_data-" + current_composition + ".csv")
        initial_data = pd.DataFrame(initial_csv).to_numpy()
        
        os.chdir(pixel_size_dir) 
        pixel_csv  = pd.read_csv(current_composition + "-Pixel size.csv")
        pixel_data = pd.DataFrame(pixel_csv).to_numpy()
        
        names_list = pixel_data[:,0].tolist()
        new_column = np.zeros(len(initial_data[:,0]))
        
        
        for i in range(len(initial_data[:,0])):
            image_name = str(initial_data[i,0]) +"-"+ initial_data[i,1] +"-"+ initial_data[i,2]  
            
            new_column[i] = pixel_data[names_list.index(image_name),1]
        
        column_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "S localization", "Comment","Image max", "Image mean", "Image median", "Center of vesicle", "Max of angular profile", "Median of angular profile", "Pixel size"]
        
        with open(path_to_output + current_composition + "-Pixel size.csv", "w", newline='') as output_file:
            df = csv.DictWriter(output_file, delimiter=',', fieldnames = column_headers)
            df.writeheader()
        
        for i in range(len(initial_data[:,0])):    
        
            vesicle_row_array = np.append(initial_data[i,0:17], new_column[i] )
            
            vesicle_row       = list(vesicle_row_array)
        
            with open(path_to_output+"\\" + current_composition + "-Pixel size.csv", "a", newline='') as output_file:
                   
                writer = csv.writer(output_file)
                writer.writerow(vesicle_row)
                
                
if Plot_size_distribution == True:
    
    ## Per composition
    for j in range(len(compositions)):
    
    
        current_composition = compositions[j]
        
        combined_csv  = pd.read_csv(path_to_output+"\\" + current_composition + "-Pixel size.csv")
        combined_data = pd.DataFrame(combined_csv).to_numpy()
        
        all_radi_px      = combined_data[:,6]
        all_radi_microns = all_radi_px*combined_data[:,17]
        
        # ## In pixel:
        # plt.figure(dpi= 150)
        # plt.title(current_composition + " - " + Septins)
        # sns.histplot(all_radi_px, color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
        # plt.xlabel("Vesicle radius (px)")
        # # plt.xlim(-5,40)
        
        ## In microns:
        plt.figure(dpi= 150)
        plt.title(current_composition + " - " + Septins)
        sns.histplot(all_radi_microns, color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
        plt.xlabel("Vesicle radius (\u03BCm)")
        plt.xlim(2,18)
        
        
 
if Size_loc_correletion == True:
    
        
        ## Per composition
        for j in range(len(compositions)):
        
        
            current_composition = compositions[j]
            
            combined_csv  = pd.read_csv(path_to_output+"\\" + current_composition + "-Pixel size.csv")
            combined_data = pd.DataFrame(combined_csv).to_numpy()
            
            all_radi_px      = combined_data[:,6]
            all_radi_microns = all_radi_px*combined_data[:,17]
        
            all_medians      = combined_data[:,16]
            all_centres      = combined_data[:,14]
            
            localization = np.zeros(len(all_medians))
            localization_classification = np.zeros(len(all_medians))
            
            for i in range(len(all_medians)):
                localization[i]     = np.clip((float(all_medians[i])-float(all_centres[i]))/float(all_medians[i]),0,None)
                
                if localization[i] > 1/6:   # corresponds to signal on the membrane > by 20% of the signal at the centre
                    
                    localization_classification[i] = 1
                else:
                    localization_classification[i] = 0
                        
           
            # plt.figure(dpi= 150)
            # plt.title(current_composition + " - " + Septins)
            # plt.scatter(all_radi_microns, localization , color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
            # plt.xlabel("Vesicle radius (\u03BCm)")
            # plt.ylabel("Localization")
            # plt.xlim(2,18)
        
            plt.figure(dpi= 150)
            plt.title(current_composition + " - " + Septins)
            plt.scatter(all_radi_microns, localization_classification , color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
            plt.xlabel("Vesicle radius (\u03BCm)")
            plt.ylabel("Localization")
            plt.xlim(2,18)
            
            
        
        
        
        
        