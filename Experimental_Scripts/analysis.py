# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:29:35 2022

@author: Katerina

Script for the statistic analysis of the protein localization on the membrane.

To be used after running main.py.
 
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import seaborn as sns
import csv

#---------------------FUNCTIONS------------------------------------------------

def find_csv_info(path_to_script):
    """
    Finds the experiment and image information of all the data sets to be analyzed,
    stored in the "output\\To_Analyze" folder of the script path. Within that folder
    files corresponding to different membrane compositions should be stored
    in seperate folders named after their composition. In turn, within each subfolder 
    it is assumed that the csv files are each in a folder named in the format:
    "experiment_date-experiment_name-image_name"
    Note: csv file names should be of the format: 
          "experiment_date-experiment_name-image_name-Output.csv"
         
    Input
    -----
    path_to_script : str
        The path to the folder containing the data to be analyzed.
    
    Returns
    -------
    compositions : np.ndarray
        1D array of strings corresponding to the membrane compositions that are
        analyzed. 
    
    num_subfolders : np.ndarray
        1D array with the number of projects available for each composition.
        Important for separating the corresponding information stored in the 
        exp_info_all_sets array. 
    
    exp_info_all_sets : np.ndarray
        All the information of all the images analyzed. 
        Note: Files of different compositions are simply stored sequentially.
        Information of different sets are stored vertically, while horizontally:
        exp_info_all_sets[:,0] is the experiment date
        exp_info_all_sets[:,1] is the experiment name
        exp_info_all_sets[:,2] is the name of the image to be analyzed
        exp_info_all_sets[:,3] is the full representative name by combining the previous information.
        
    """
    path_to_data_folder = path_to_script + "\\output\\To_Analyze"
    os.chdir(path_to_data_folder)
    exp_info_all_sets = np.zeros((4), dtype = str)
    
    set_of_data  = 0
    compositions   = np.zeros((1), dtype = str)
    num_subfolders = np.zeros((1), dtype = str)
    exp_info_all_sets = np.zeros((4), dtype = str)
    
    for f in os.listdir():
     
        if set_of_data == 0:
            compositions = [f]
        else:
            compositions = np.append(compositions, f)
        
        os.chdir(path_to_data_folder + "\\" + f)
        
        if set_of_data == 0:
            num_subfolders = [len(os.listdir())]
        else:
            num_subfolders = np.append(num_subfolders, len(os.listdir()) )
            
        for d in os.listdir():
            
            name = d
            
            exp_info_single_set = name.split("-")
            exp_info_single_set.append(exp_info_single_set[0] + "-" + exp_info_single_set[1] + "-" + exp_info_single_set[2])
         
            if set_of_data == 0:
                exp_info_all_sets = np.array(exp_info_single_set)
            else:
                exp_info_all_sets = np.vstack((exp_info_all_sets, exp_info_single_set))
            
            set_of_data += 1
    
    os.chdir(path_to_script)
    
    return compositions, num_subfolders, exp_info_all_sets
#----------------------------------------------------------------

#------------------------- INPUT --------------------------------

## Specify path to the data directory and experiment information:
path_to_script      = "D:\\Personal\\TU Delft\\Thesis\\Intensity profiles"
path_to_data_folder = path_to_script + "\\output\\To_Analyze"

## Defining which proteins are present (True) or absent (False):
Septin              = True       
Actin               = False  ## NOT YET ACOMMODATED!!

#----------------------------------------------------------------

#------------------------ PREPARATION ---------------------------

## Rearranging input:
proteins_present    = np.array((Septin, Actin), dtype = bool) ## NOT YET ACOMMODATED!!

## Finding experiment and image information for all data sets to be analyzed:
compositions, num_subfolders, exp_info_all_sets   = find_csv_info(path_to_script)

## Creating file to store all "OK" vesicles for each composition:
column_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "S localization", "Comment"]

for c in compositions:
    
    with open(path_to_script + "\\output" +"\\Auto_filtered_data-" + c + ".csv", "w", newline='') as collective_file:
        df = csv.DictWriter(collective_file, delimiter=',', fieldnames = column_headers)
        df.writeheader()    


# #----------------------------------------------------------------

# #--------------------------- MAIN -------------------------------

## Iterate for each compositions and create file with collective statistics:

exp_info_to_analyze = exp_info_all_sets
    
for c in range(len(compositions)):
    
    ## Focusing in the experiment info of a specific composition:
    exp_info_same_composition = exp_info_to_analyze[:num_subfolders[c],:]
    exp_info_to_analyze       = np.delete(exp_info_to_analyze, slice(num_subfolders[c]), axis=0)  # Preparation for the next iteration
    
    j = 0
    
    ## Iterating over all files of the same composition:
    for n in exp_info_same_composition[:,3]:
        
        os.chdir(path_to_data_folder + "\\" + compositions[c] + "\\" + n)
        
        output_csv  = pd.read_csv(exp_info_same_composition[j,3] + "-Output.csv")
        output = pd.DataFrame(output_csv).to_numpy()
        
        for row in range(len(output[:,10,])):
            
            ## Collecting all "OK" vesicles:
            if output[row,10] == "OK":
                
                vesicle_row = output[row,:]
                
                with open(path_to_script + "\\output" +"\\Auto_filtered_data-" + compositions[c] + ".csv", "a", newline='') as collective_file:
                           
                    writer = csv.writer(collective_file)
                    writer.writerow(vesicle_row)               
        
        j += 1
        
os.chdir(path_to_script)

set_of_data   = 0
num_comp_data = np.zeros((1), dtype = str)
all_data      = np.zeros((1), dtype = str)
 
for c in range(len(compositions)):
    
    ## Open collective file for each composition and collect the localization values:
    
    os.chdir(path_to_script + "\\output")
    
    collective_csv  = pd.read_csv("Auto_filtered_data-" + compositions[c] + ".csv")
    collective_data = pd.DataFrame(collective_csv).to_numpy()
    
    if set_of_data == 0:
        num_comp_data = len(collective_data[:,0])
        all_data      = collective_data[:,9]
        
    else:
        num_comp_data = np.append(num_comp_data, len(collective_data[:,0]))
        all_data      = np.append(all_data, collective_data[:,9])
    set_of_data += 1
 

separator = np.cumsum(num_comp_data)
print(separator)
    

# boxplot_data = {compositions[0] + "\n N$_{vesicles}$ =" + str(num_comp_data[2]) : all_data[:separator[0]], compositions[2]+ "\n N$_{vesicles}$ =" + str(num_comp_data[3]): all_data[separator[1]:separator[2]], compositions[0]+ "\n N$_{vesicles}$ =" + str(num_comp_data[1]): all_data[:separator[0]], compositions[3] + "\n N$_{vesicles}$ =" + str(num_comp_data[3]): all_data[separator[2]:separator[3]], compositions[4] + "\n N$_{vesicles}$ =" + str(num_comp_data[5]): all_data[separator[3]:separator[4]], compositions[5]: all_data[separator[4]:separator[5]]}#, 'DEF': [34.541, 34.748, 34.482]}
 
## Creating boxplot:

boxplot_data = {compositions[1] + "\n N$_{vesicles}$ =" + str(num_comp_data[1]) : all_data[separator[0]:separator[1]], compositions[2]+ "\n N$_{vesicles}$ =" + str(num_comp_data[2]): all_data[separator[1]:separator[2]], compositions[0]+ "\n N$_{vesicles}$ =" + str(num_comp_data[0]): all_data[:separator[0]], compositions[3] + "\n N$_{vesicles}$ =" + str(num_comp_data[3]): all_data[separator[2]:separator[3]], compositions[4] + "\n N$_{vesicles}$ =" + str(num_comp_data[4]): all_data[separator[3]:separator[4]], compositions[5]+ "\n N$_{vesicles}$ =" + str(num_comp_data[5]): all_data[separator[4]:separator[5]]}#, 'DEF': [34.541, 34.748, 34.482]}

fig, ax = plt.subplots(figsize=(14,6), dpi=150)
plt.title("Membrane-septin hexamer co-localization")
ax.boxplot(boxplot_data.values(), sym="")
ax.set_xticklabels(boxplot_data.keys())
plt.ylabel("Localization")
# plt.ylim(-0.2,10)


# plt.figure(dpi= 150)
# plt.title("Membrane-septin octamer co-localization")
# sns.histplot(all_data[:separator[0]], color="purple", alpha=0.5, edgecolor='none', label=compositions[0]+"\n N$_{vesicles}$ = " + str(num_comp_data[0]))
 
# if len(compositions) > 1:
#     sns.histplot(all_data[separator[0]:separator[1]], color="blue", alpha=0.5, edgecolor='none', label=compositions[1]+"\n N$_{vesicles}$ = " + str(num_comp_data[1]))

# if len(compositions) > 2:
#     sns.histplot(all_data[separator[1]:separator[2]], color="green", alpha=0.5, edgecolor='none', label=compositions[2]+"\n N$_{vesicles}$ = " + str(num_comp_data[2]))

# if len(compositions) > 3:
#     sns.histplot(all_data[separator[2]:separator[3]], color="yellow", alpha=0.5, edgecolor='none', label=compositions[3]+"\n N$_{vesicles}$ = " + str(num_comp_data[3]))

# if len(compositions) > 4:
#     sns.histplot(all_data[separator[3]:separator[4]], color="orange", alpha=0.5, edgecolor='none', label=compositions[4]+"\n N$_{vesicles}$ = " + str(num_comp_data[4]))

# if len(compositions) > 5:
#     sns.histplot(all_data[separator[4]:separator[5]], color="red", alpha=0.5, edgecolor='none', label=compositions[5]+"\n N$_{vesicles}$ = " + str(num_comp_data[5]))

# # plt.xlabel("Localization")
# plt.xlim(-0.2,10)
# plt.legend()
# plt.show()
     
    
 


