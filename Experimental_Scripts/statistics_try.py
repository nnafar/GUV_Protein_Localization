# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:54:07 2023

@author: Katerina
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:46:36 2023

@author: Katerina

Script for the statistical analysis of the protein localization on the membrane.

To be used after running main.py.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import seaborn as sns
import csv
from scipy.stats import mannwhitneyu
from statannot import add_stat_annotation
from scipy import stats

## Parameters:-----------------------------------------------------------------

# Files........................................................................

Septins       = "Hexamers"    
#Septins      = "Octamers"

#Data  = "Prior cluster correction"
#Data  = "After cluster correction"
#Data  = "High rsd removed"
Data  = "New definitions"

Data_version = ""
if Septins == "Octamers" and Data == "After cluster correction":
    Data_version = "New"
    #Data_version = "Old"

    
#..............................................................................

# What to calculate............................................................
Average           = False
Median            = False
Save              = False
Classification    = False
Distrib_per_sample= True
Boxplots          = False
High_rsd          = False
Violin_plot       = False
Size_distribution = False
MWU               = False
#..............................................................................


## Specify path to the data directory and experiment information:
path_to_script      = "D:\\Personal\\TU Delft\\Thesis\\Intensity profiles"

if Data == "After cluster correction":
    path_to_data_hexamers = path_to_script + "\\output\\After cluster detection\\Hexamers"
    
    if Data_version == "Old":
        path_to_data_octamers = path_to_script + "\\output\\After cluster detection\\Octamers\\Analysis old"
    elif Data_version == "New":
        path_to_data_octamers = path_to_script + "\\output\\After cluster detection\\Octamers\\Analysis new"

elif Data == "Prior cluster correction":
    path_to_data_octamers = path_to_script + "\\output\\Prior to cluster correction in localization quantification\\Octamers"
    path_to_data_hexamers = path_to_script + "\\output\\Prior to cluster correction in localization quantification\\Hexamers"

elif Data == "High rsd removed":
    path_to_data_octamers = path_to_script + "\\output\\After cluster detection\\Octamers\\No high_rsd"
    path_to_data_hexamers = path_to_script + "\\output\\After cluster detection\\Hexamers\\No high_rsd"

elif Data == "New definitions":
    path_to_data_octamers = path_to_script + "\\output\\New definitions\\Octamers"
    path_to_data_hexamers = path_to_script + "\\output\\New definitions\\Hexamers"
    
## Reading files:

compositions = ["100% DOPC","0% DOPS, 5% PIP","20% DOPS, 0% PIP","20% DOPS, 1% PIP","20% DOPS, 3% PIP","20% DOPS, 5% PIP"]

set_of_data   = 0
num_comp_data = np.zeros((1), dtype = str)
all_data      = np.zeros((1), dtype = str)
 
for c in range(len(compositions)):
    
    ## Open collective file for each composition and collect the localization values:
    
    if Septins == "Octamers":
        os.chdir(path_to_data_octamers)
    elif Septins == "Hexamers":
        os.chdir(path_to_data_hexamers)
    
    collective_csv  = pd.read_csv("Auto_filtered_data-" + compositions[c] + ".csv")
    collective_data = pd.DataFrame(collective_csv).to_numpy()
    
    if set_of_data == 0:
        num_comp_data = len(collective_data[:,0])
        all_dates     = collective_data[:,0]
        all_projects  = collective_data[:,1]
        all_images    = collective_data[:,2]
        all_id        = collective_data[:,3]
        all_x         = collective_data[:,4]
        all_y         = collective_data[:,5]
        all_radi      = collective_data[:,6]
        all_m_back    = collective_data[:,7]
        all_s_back    = collective_data[:,8]
        all_data      = collective_data[:,9]
        all_comments  = collective_data[:,10]
        all_image_max    = collective_data[:,11]
        all_image_mean   = collective_data[:,12]
        all_image_median = collective_data[:,13]
        all_centres   = collective_data[:,14]
        all_vesicle_max  = collective_data[:,15]
        all_medians   = collective_data[:,16]
        
    else:
        num_comp_data = np.append(num_comp_data, len(collective_data[:,0]))
        all_dates     = np.append(all_dates, collective_data[:,0])
        all_projects  = np.append(all_projects, collective_data[:,1])
        all_images    = np.append(all_images, collective_data[:,2])
        all_id        = np.append(all_id, collective_data[:,3])
        all_x         = np.append(all_x, collective_data[:,4])
        all_y         = np.append(all_y, collective_data[:,5])
        all_radi      = np.append(all_radi, collective_data[:,6])
        all_m_back    = np.append(all_m_back, collective_data[:,7])
        all_s_back    = np.append(all_s_back, collective_data[:,8])
        all_data      = np.append(all_data, collective_data[:,9])
        all_comments  = np.append(all_comments, collective_data[:,10])
        all_image_max    = np.append(all_image_max, collective_data[:,11])
        all_image_mean   = np.append(all_image_mean, collective_data[:,12])
        all_image_median = np.append(all_image_median, collective_data[:,13])
        all_centres      = np.append(all_centres, collective_data[:,14])
        all_vesicle_max  = np.append(all_vesicle_max, collective_data[:,15])
        all_medians   = np.append(all_medians, collective_data[:,16])
        
        
        
    set_of_data += 1
    
    
separator = np.cumsum(num_comp_data)

## New definition:
    
## For new definition:
all_data_float = np.zeros(len(all_medians))

for i in range(len(all_data)):
    all_data_float[i] = np.clip((float(all_medians[i])-float(all_centres[i]))/float(all_medians[i]),0,None)
    all_data = all_data_float


#..............................................................................
if Average == True:
    average_loc = np.zeros(len(compositions))
    
    average_loc[0] = np.average(all_data[:separator[0]])
    
    for c in range(len(compositions)-1):
        average_loc[c+1] =  np.average(all_data[separator[c]:separator[c+1]])
        
    print("average_loc",average_loc)
#..............................................................................

#..............................................................................
if Median == True:
    median_loc = np.zeros(len(compositions))
    
    median_loc[0] = np.median(all_data[:separator[0]])
    
    for c in range(len(compositions)-1):
        median_loc[c+1] =  np.median(all_data[separator[c]:separator[c+1]])
        
    print("median_loc",median_loc)
#..............................................................................

#..............................................................................

if Save==True:
    
    column_headers = [compositions[0],compositions[1],compositions[2],compositions[3],compositions[4],compositions[5]]#,"Image max","Image median", "Image mean", "Vesicle centre", "Vesicle angular prof. max", "Vesicle angular prof. median"]


    with open(path_to_script + "\\output\\New definitions\\Average and median localization - "+ Septins + ".csv", "w", newline='') as image_stuff:
        df = csv.DictWriter(image_stuff, delimiter=',', fieldnames = column_headers)
        df.writeheader() 

    
        
    vesicle_row_one = average_loc
    vesicle_row_two = list(median_loc) 
    
    with open(path_to_script + "\\output\\New definitions\\Average and median localization - "+ Septins + ".csv", "a", newline='') as image_stuff:
               
        writer = csv.writer(image_stuff)
        writer.writerow(vesicle_row_one)   
        writer.writerow(vesicle_row_two)
#..............................................................................

if Classification == True:
    ## NOTE:saves the onew with the value of the new definition. That's why threshold is 1/6 corresponding to 1.2
    percentage_loc = np.zeros(len(compositions))
    
    
    data_0 = all_data[:separator[0]]
    data_1 = all_data[separator[0]:separator[1]]
    data_2 = all_data[separator[1]:separator[2]]
    data_3 = all_data[separator[2]:separator[3]]
    data_4 = all_data[separator[3]:separator[4]]
    data_5 = all_data[separator[4]:separator[5]]     
    
    column_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "S localization", "Comment", "Image max", "Image mean", "Image median", "Center of vesicle", "Max of angular profile", "Median of angular profile"]

    for i in range(len(compositions)):
        
        with open(path_to_script + "\\output\\New definitions\\Localized vesicles\\Localized vesicles - " + compositions[i] + " - " + Septins + ".csv", "w", newline='') as loc_ves:
            df = csv.DictWriter(loc_ves, delimiter=',', fieldnames = column_headers)
            df.writeheader() 
    
    
    count_loc = 0
    for i in range(len(data_0)):
        if data_0[i] >= 1/6:
            count_loc += 1
            
            vesicle_row = all_dates[i], all_projects[i], all_images[i], all_id[i], all_x[i], all_y[i], all_radi[i], all_m_back[i], all_s_back[i], all_data[i], all_comments[i], all_image_max[i], all_image_mean[i], all_image_median[i], all_centres[i], all_vesicle_max[i], all_medians[i] 
            
            with open(path_to_script + "\\output\\New definitions\\Localized vesicles\\Localized vesicles - " + compositions[0] + " - " + Septins + ".csv", "a", newline='') as loc_ves:
                       
                writer = csv.writer(loc_ves)
                writer.writerow(vesicle_row)
                
    percentage_loc[0] = count_loc / len(data_0) * 100
    
    count_loc = 0
    for i in range(len(data_1)):
        if data_1[i] >= 1/6:
            count_loc += 1
            
            vesicle_row = all_dates[separator[0]+i], all_projects[separator[0]+i], all_images[separator[0]+i], all_id[separator[0]+i], all_x[separator[0]+i], all_y[separator[0]+i], all_radi[separator[0]+i], all_m_back[separator[0]+i], all_s_back[separator[0]+i], all_data[separator[0]+i], all_comments[separator[0]+i], all_image_max[separator[0]+i], all_image_mean[separator[0]+i], all_image_median[separator[0]+i], all_centres[separator[0]+i], all_vesicle_max[separator[0]+i], all_medians[separator[0]+i] 
            
            with open(path_to_script + "\\output\\New definitions\\Localized vesicles\\Localized vesicles - " + compositions[1] + " - " + Septins + ".csv", "a", newline='') as loc_ves:
                       
                writer = csv.writer(loc_ves)
                writer.writerow(vesicle_row)
                
    percentage_loc[1] = count_loc / len(data_1) * 100
    
    count_loc = 0
    for i in range(len(data_2)):
        if data_2[i] >= 1/6:
            count_loc += 1
            
            vesicle_row = all_dates[separator[1]+i], all_projects[separator[1]+i], all_images[separator[1]+i], all_id[separator[1]+i], all_x[separator[1]+i], all_y[separator[1]+i], all_radi[separator[1]+i], all_m_back[separator[1]+i], all_s_back[separator[1]+i], all_data[separator[1]+i], all_comments[separator[1]+i], all_image_max[separator[1]+i], all_image_mean[separator[1]+i], all_image_median[separator[1]+i], all_centres[separator[1]+i], all_vesicle_max[separator[1]+i], all_medians[separator[1]+i] 
            
            with open(path_to_script + "\\output\\New definitions\\Localized vesicles\\Localized vesicles - " + compositions[2] + " - " + Septins + ".csv", "a", newline='') as loc_ves:
                       
                writer = csv.writer(loc_ves)
                writer.writerow(vesicle_row)
                
    percentage_loc[2] = count_loc / len(data_2) * 100
    
    count_loc = 0
    for i in range(len(data_3)):
        if data_3[i] >= 1/6:
            count_loc += 1
            
            vesicle_row = all_dates[separator[2]+i], all_projects[separator[2]+i], all_images[separator[2]+i], all_id[separator[2]+i], all_x[separator[2]+i], all_y[separator[2]+i], all_radi[separator[2]+i], all_m_back[separator[2]+i], all_s_back[separator[2]+i], all_data[separator[2]+i], all_comments[separator[2]+i], all_image_max[separator[2]+i], all_image_mean[separator[2]+i], all_image_median[separator[2]+i], all_centres[separator[2]+i], all_vesicle_max[separator[2]+i], all_medians[separator[2]+i] 
            
            with open(path_to_script + "\\output\\New definitions\\Localized vesicles\\Localized vesicles - " + compositions[3] + " - " + Septins + ".csv", "a", newline='') as loc_ves:
                       
                writer = csv.writer(loc_ves)
                writer.writerow(vesicle_row)
                
    percentage_loc[3] = count_loc / len(data_3) * 100
    
    count_loc = 0
    for i in range(len(data_4)):
        if data_4[i] >= 1/6:
            count_loc += 1
            
            vesicle_row = all_dates[separator[3]+i], all_projects[separator[3]+i], all_images[separator[3]+i], all_id[separator[3]+i], all_x[separator[3]+i], all_y[separator[3]+i], all_radi[separator[3]+i], all_m_back[separator[3]+i], all_s_back[separator[3]+i], all_data[separator[3]+i], all_comments[separator[3]+i], all_image_max[separator[3]+i], all_image_mean[separator[3]+i], all_image_median[separator[3]+i], all_centres[separator[3]+i], all_vesicle_max[separator[3]+i], all_medians[separator[3]+i] 
            
            with open(path_to_script + "\\output\\New definitions\\Localized vesicles\\Localized vesicles - " + compositions[4] + " - " + Septins + ".csv", "a", newline='') as loc_ves:
                       
                writer = csv.writer(loc_ves)
                writer.writerow(vesicle_row)
                
    percentage_loc[4] = count_loc / len(data_4) * 100
    
    count_loc = 0
    for i in range(len(data_5)):
        if data_5[i] >= 1/6:
            count_loc += 1
            
            vesicle_row = all_dates[separator[4]+i], all_projects[separator[4]+i], all_images[separator[4]+i], all_id[separator[4]+i], all_x[separator[4]+i], all_y[separator[4]+i], all_radi[separator[4]+i], all_m_back[separator[4]+i], all_s_back[separator[4]+i], all_data[separator[4]+i], all_comments[separator[4]+i], all_image_max[separator[4]+i], all_image_mean[separator[4]+i], all_image_median[separator[4]+i], all_centres[separator[4]+i], all_vesicle_max[separator[4]+i], all_medians[separator[4]+i] 
            
            with open(path_to_script + "\\output\\New definitions\\Localized vesicles\\Localized vesicles - " + compositions[5] + " - " + Septins + ".csv", "a", newline='') as loc_ves:
                       
                writer = csv.writer(loc_ves)
                writer.writerow(vesicle_row)
                
    percentage_loc[5] = count_loc / len(data_5) * 100

    print(percentage_loc)

    column_headers = [compositions[0],compositions[1],compositions[2],compositions[3],compositions[4],compositions[5]]#,"Image max","Image median", "Image mean", "Vesicle centre", "Vesicle angular prof. max", "Vesicle angular prof. median"]

    with open(path_to_script + "\\output\\New definitions\\Classification_localization_percetages - "+ Septins + ".csv", "w", newline='') as loc_percent:
        df = csv.DictWriter(loc_percent, delimiter=',', fieldnames = column_headers)
        df.writeheader() 

    vesicle_row = percentage_loc
    
    with open(path_to_script + "\\output\\New definitions\\Classification_localization_percetages - "+ Septins + ".csv", "a", newline='') as loc_percent:
               
        writer = csv.writer(loc_percent)
        writer.writerow(vesicle_row)   
        
    percentages_localization = pd.read_csv(path_to_script + "\\output\\New definitions\\Class_loc_percetages - "+ Septins + ".csv")
    percentages_localization_data = pd.DataFrame(percentages_localization).to_numpy()
    
    
    
    fig, ax = plt.subplots(figsize=(10.5,6), dpi=150)
    ax = sns.barplot(x='Compositions', y='Percentages', hue=' ',data=percentages_localization)#, palette=["forestgreen", "firebrick"]) 
    plt.ylabel("Vesicles (%)")
    plt.xlabel(None)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', fontsize=14)
    
    
    
    
#..............................................................................
 
if Distrib_per_sample == True:
    
    # For info from current script:
    select_number          = 3
    selected_composition   = compositions[select_number]
    selected_data_medians  = all_medians[separator[select_number-1]:separator[select_number]]
    selected_data_centres  = all_centres[separator[select_number-1]:separator[select_number]]
    selected_data_dates    = all_dates[separator[select_number-1]:separator[select_number]]
    selected_data_projects = all_projects[separator[select_number-1]:separator[select_number]]
    
    break_point = [0]

    for i in range(len(selected_data_medians)-1):
        if selected_data_dates[i] != selected_data_dates[i+1] or selected_data_projects[i] != selected_data_projects[i+1]:
            break_point = np.append(break_point, i+1)

    print(break_point)
    
    # plt.hist(selected_data_medians, bins=40)
    # plt.hist(selected_data_centres, bins=40, color = "orange")
    # plt.show()
 
    # # x = np.arange(max(selected_data_centres))
    # # y = 1 * x
    
    # plt.scatter(selected_data_centres, (selected_data_medians-selected_data_centres)/selected_data_medians, marker='.')
    # plt.title(compositions[select_number] + " - " + Septins)
    # plt.xlabel("xlabel, kwargs")
    # plt.xlabel("Septin signal at the centre")
    # plt.ylabel("Septin signal at the membrane")
    # # plt.plot(x,y)
    # # plt.ylim(0,70)
    # # plt.xlim(0,25)
    # plt.show()
    
    ## For info from only localized vesicles:
    # select_number          = 3
    # selected_composition   = compositions[select_number]
    # loc_collected_csv  = pd.read_csv(path_to_script + "\\output\\New definitions\\Localized vesicles\\Localized vesicles - " + compositions[select_number] + " - " + Septins + ".csv")
    # loc_collected_data = pd.DataFrame(loc_collected_csv).to_numpy()
    
    # selected_data_medians  = loc_collected_data[:,16]
    # selected_data_centres  = loc_collected_data[:,14]
    # selected_data_dates    = loc_collected_data[:,0]
    # selected_data_projects = loc_collected_data[:,1]
    
    # break_point = [0]

    for i in range(len(selected_data_medians)-1):
        if selected_data_dates[i] != selected_data_dates[i+1] or selected_data_projects[i] != selected_data_projects[i+1]:
            break_point = np.append(break_point, i+1)

    print(break_point)
    
    # plt.title("Distributions - " + compositions[select_number] + " - " + Septins)
    # plt.hist(selected_data_medians, bins=50, color = "tab:blue", label="Median of signal at membrane")
    # plt.hist(selected_data_centres, bins=50, color = "orange", label="Mean signal at centre")
    # plt.legend()
    # plt.show()
    
    float_1 = np.zeros(len(selected_data_centres[break_point[2]:]))
    float_2 = np.zeros(len(selected_data_medians[break_point[2]:]))
    
    for i in range(len(selected_data_centres[break_point[2]:])):
        float_1[i] = float(selected_data_centres[break_point[2]+i])
        float_2[i] = float(selected_data_medians[break_point[2]+i])
        
    z = np.polyfit(float_1, float_2, 1)
    print(z)
    
    print(np.mean(selected_data_centres))
    x = np.arange(max(selected_data_centres))
    y = z[0] * x +z[1]
    
    fig, ax = plt.subplots(figsize=(10.5,6), dpi=150)
    plt.scatter(selected_data_centres[break_point[2]:], selected_data_medians[break_point[2]:], marker='.')
    plt.title(compositions[select_number] + " - " + Septins)
    plt.xlabel("xlabel, kwargs")
    plt.xlabel("Average septin signal at the centre")
    plt.ylabel("Median septin signal at membrane")
    plt.plot(x,y,'black', label='Ax + c,  A= ' + str(round(z[0],2)) + ', C = ' + str(round(z[1],2)))
    plt.ylim(2,25)
    plt.xlim(0,10)
    plt.legend()
    
    # fig, ax = plt.subplots(figsize=(10.5,6), dpi=150)
    # plt.scatter(selected_data_centres[break_point[2]:], selected_data_medians[break_point[2]:]-selected_data_centres[break_point[2]:], marker='.')
    # plt.title(compositions[select_number] + " - " + Septins)
    # plt.xlabel("xlabel, kwargs")
    # plt.xlabel("Average septin signal at the centre")
    # plt.ylabel("Median septin signal at membrane")
    # plt.plot(x,y)
    # plt.ylim(2,25)
    # # plt.xlim(0,25)
    
    
    
#..............................................................................
if High_rsd == True:
    
    marked_rsd = np.zeros(len(compositions))
    remaining  = np.zeros(len(compositions))
    
    count_rsd = np.zeros(len(compositions))
    for i in range(0,separator[0]):
        if all_comments[i] == "high_rsd":
            count_rsd[0] += 1
    marked_rsd[0] = count_rsd[0]/num_comp_data[0]
    remaining[0]  = num_comp_data[0] - count_rsd[0]
    for c in range(len(compositions)-1):
        
        for i in range(separator[c],separator[c+1]):
            if all_comments[i] == "high_rsd":
                count_rsd[c+1] += 1
        marked_rsd[c+1] = count_rsd[c+1]/num_comp_data[c+1]
        remaining[c+1]  = num_comp_data[c+1] - count_rsd[c+1]
    print("marked_rsd",marked_rsd)
    print("remaining",remaining)

#..............................................................................

#..............................................................................
if Boxplots == True:
    
    if MWU == True:
        
        ## For new definition:
        all_data_float = np.zeros(len(all_medians))
        
        for i in range(len(all_data)):
            all_data_float[i] = np.clip((float(all_medians[i])-float(all_centres[i]))/float(all_medians[i]),0,None)
            all_data = all_data_float
        
        
        ## For original definition:
        # all_data_float = np.zeros(len(all_data))
        
        # for i in range(len(all_data)):
        #     all_data_float[i] = float(all_data[i])
        
        
        data_0 = all_data_float[:separator[0]]
        data_1 = all_data_float[separator[0]:separator[1]]
        data_2 = all_data_float[separator[1]:separator[2]]
        data_3 = all_data_float[separator[2]:separator[3]]
        data_4 = all_data_float[separator[3]:separator[4]]
        data_5 = all_data_float[separator[4]:separator[5]]        
        
        data_list = [data_0, data_1, data_2, data_3, data_4, data_5]
        
        pvalues = []
        
        ls = list(range(1, 6 + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
        
        # combinations = [(1, 6), (1, 5), (1, 4), (5, 6), (1, 3), (4, 5), (1, 2), (3, 4), (2,6), (2,5), (2,4), (2,3)]#, (3, 5), (4, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        # combinations = [(1,6),(1,5),(5,6),(1,4),(4,6),(1,3),(3,6),(1,2),(2,6),(2,5),(2,4),(4,5),(2,3),(3,5),(3,4)]
        combinations = [(2,6),(1,3),(3,4),(4,6),(1,2),(2,3), (4,5),(5,6)]
        
        for combination in combinations:
            x = data_list[combination[0]-1]
            y = data_list[combination[1]-1]
        
            pvalues = np.append(pvalues, mannwhitneyu(x,y,alternative="two-sided").pvalue)
    # Colour of the mean lines
    meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')
    
    boxplot_data = {compositions[0] + "\n N$_{vesicles}$ =" + str(num_comp_data[0]): all_data[:separator[0]], compositions[1] + "\n N$_{vesicles}$ =" + str(num_comp_data[1]) : all_data[separator[0]:separator[1]], compositions[2] + "\n N$_{vesicles}$ =" + str(num_comp_data[2]) : all_data[separator[1]:separator[2]], compositions[3] + "\n N$_{vesicles}$ =" + str(num_comp_data[3]) : all_data[separator[2]:separator[3]] , compositions[4] + "\n N$_{vesicles}$ =" + str(num_comp_data[4]): all_data[separator[3]:separator[4]], compositions[5]+ "\n N$_{vesicles}$ =" + str(num_comp_data[5]): all_data[separator[4]:separator[5]]}
   
    fig, ax = plt.subplots(figsize=(14,8), dpi=150)
    if Data_version == "":
        plt.title(Septins + " localization on the GUV membrane" + " (" + Data + ")")
    else:
        plt.title(Septins + " localization on the GUV membrane" + " (" + Data + "-" + Data_version + ")")
    bp = ax.boxplot(boxplot_data.values(), sym="", showmeans = "True", meanprops = meanpointprops)
    
    
    #add_stat_annotation(ax, data=data_annotate, box_pairs=pairs,test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
    ax.set_xticklabels(boxplot_data.keys())
    plt.ylabel("Localization")
    # Colour of the median lines
    plt.setp(bp['medians'], color='r')

    
    colors = ['b.','y.','g.','r.','m.','b.']
    ## Octamers no_rsd
    # thresholds_up_all= [1.2,1.3,2,5.5,4.3,3]#[item.get_ydata()[1] for item in bp['whiskers']]
    # thresholds_down_all= [0.3,0,0,0,0,0]
    
    ## Hexamers no_rsd
    # thresholds_up_all= [0.85,5,1,3,3.3,3.7]#[item.get_ydata()[1] for item in bp['whiskers']]
    # thresholds_down_all= [0.35,1,0,0,0,0]
    
    ## Hexamers with the high rsd
    # thresholds_up_all= [0.95,5.8,2.3,4.5,4.1,3.9]#[item.get_ydata()[1] for item in bp['whiskers']]
    # thresholds_down_all= [0.4,0,0,0,0,0]
    
    ## Octamers with the high rsd
    # thresholds_up_all= [3,2.5,3.7,5.3,4.5,4.7]#[item.get_ydata()[1] for item in bp['whiskers']]
    # thresholds_down_all= [0,0,0,0,0,0]
    
    ## Hexamers with new definition with negatives
    # thresholds_up_all= [1,1,1,1,1,1]#[item.get_ydata()[1] for item in bp['whiskers']]
    # thresholds_down_all= [-3,-3,-3,-3,-3,-3]
    
    ## Octamers with new definition with negatives
    # thresholds_up_all= [1,1,1,1,1,1]#[item.get_ydata()[1] for item in bp['whiskers']]
    # thresholds_down_all= [-3,-3,-3,-3,-3,-3]
    
    # All with new definition
    thresholds_up_all= [1,1,1,1,1,1]#[item.get_ydata()[1] for item in bp['whiskers']]
    thresholds_down_all= [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1]
                          
    for i in range(6):
        threshold_up   = thresholds_up_all[i]
        threshold_down = thresholds_down_all[i]
        y_all = list(boxplot_data.values())[i]
        
        y = []
        for j in range(len(y_all)):
            if y_all[j] < threshold_up and y_all[j] > threshold_down:
                
                y = np.append(y, y_all[j])
        x = np.random.normal(1+i, 0.04, size=len(y))
        plt.plot(x, y, colors[i], alpha=0.2)
    #ax.set_ylim(0,20)

    if MWU == True:
        # Get the y-axis limits
        bottom, top = ax.get_ylim()
        y_range = top - bottom
        
        index_position = 0
        index          = 0
        line_break = 1
        time = 0
        for combination in combinations:
            
            # Columns corresponding to the datasets of interest
            x1 = combination[0]
            x2 = combination[1]
            # What level is this bar among the bars above the plot?
            if index_position >=5 and index_position <=7 and time == 0:
                line_break = 1
                time += 1
            elif index_position >=5 and index_position <=7 and time == 1:
                line_break = 2
                time += 2
            elif index_position >7 and  time == 2:
                line_break = 2
                time += 1
            elif index_position >7 and time == 3:
                line_break =2
                time += 1
            elif index_position >7 and time == 4:
                line_break =3
                
            if combination[0] != line_break:
                index_position = index_position -1
            # if index_position>5:
            #     index_position = index_position +1
            # level = 15 - 1.6*index_position
            level = 3 - 1.6*index_position
            # Plot the bar
            bar_height = (y_range * 0.07 * level) + top
            bar_tips = bar_height - (y_range * 0.025)
            plt.plot([x1+0.05, x1+0.05, x2-0.05, x2-0.05], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            
            if pvalues[index] < 0.0001:
                sig_symbol = '****'
            elif pvalues[index] < 0.001:
                sig_symbol = '***'
            elif pvalues[index] < 0.01:
                sig_symbol = '**'
            elif pvalues[index] < 0.05:
                sig_symbol = '*'
            elif pvalues[index] > 0.05:
                sig_symbol = 'ns'
            
            if sig_symbol == 'ns':
                text_height = bar_height + (y_range * 0.01)
            else:
                text_height = bar_height + (y_range * 0.000001)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
            
            index_position +=1
            index += 1

    
    plt.show()
    
    
#..............................................................................

if Violin_plot == True:
    boxplot_data = {compositions[0] + "\n N$_{vesicles}$ =" + str(num_comp_data[0]): all_data[:separator[0]], compositions[1] + "\n N$_{vesicles}$ =" + str(num_comp_data[1]) : all_data[separator[0]:separator[1]], compositions[2] + "\n N$_{vesicles}$ =" + str(num_comp_data[2]) : all_data[separator[1]:separator[2]], compositions[3] + "\n N$_{vesicles}$ =" + str(num_comp_data[3]) : all_data[separator[2]:separator[3]] , compositions[4] + "\n N$_{vesicles}$ =" + str(num_comp_data[4]): all_data[separator[3]:separator[4]], compositions[5]+ "\n N$_{vesicles}$ =" + str(num_comp_data[5]): all_data[separator[4]:separator[5]]}
    df = [all_data[:separator[0]], all_data[separator[0]:separator[1]], all_data[separator[1]:separator[2]], all_data[separator[2]:separator[3]], all_data[separator[3]:separator[4]], all_data[separator[4]:separator[5]]]
    fig, ax = plt.subplots(figsize=(14,6), dpi=150)
    if Data_version == "":
        plt.title("Membrane-Septin " + Septins + " co-localization" + " (" + Data + ")")
    else:
        plt.title("Membrane-Septin " + Septins + " co-localization" + " (" + Data + "-" + Data_version + ")")
    sns.violinplot(df, showmeans=True, color="lightsteelblue")
    sns.stripplot(df, jitter=True, size=3)
    ax.set_xticklabels(boxplot_data.keys())
    ax.set_ylim(0,1.25)
    plt.ylabel("Localization")
    
#..............................................................................    
    
if Size_distribution == True:
    
    ## All compositions
    plt.figure(dpi= 150)
    plt.title("All compositions - Vesicle radi distribution")
    sns.histplot(all_radi, color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
    plt.xlabel("Vesicle radi (px)")
    # plt.xlim(-5,40)
    
    ## 100% DOPC
    plt.figure(dpi= 150)
    plt.title("100% DOPC - Vesicle radi distribution")
    sns.histplot(all_radi[:separator[0]], color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
    plt.xlabel("Vesicle radi (px)")
    # plt.xlim(-5,40)
    
    ## 0% DOPS, 5% PIP2
    plt.figure(dpi= 150)
    plt.title("0% DOPS, 5% PIP2 - Vesicle radi distribution")
    sns.histplot(all_radi[separator[0]:separator[1]], color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
    plt.xlabel("Vesicle radi (px)")
    # plt.xlim(-5,40)
    
    ## 20% DOPS, 0% PIP2
    plt.figure(dpi= 150)
    plt.title("20% DOPS, 0% PIP2 - Vesicle radi distribution")
    sns.histplot(all_radi[separator[1]:separator[2]], color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
    plt.xlabel("Vesicle radi (px)")
    
    ## 20% DOPS, 1% PIP2
    plt.figure(dpi= 150)
    plt.title("20% DOPS, 1% PIP2 - Vesicle radi distribution")
    sns.histplot(all_radi[separator[2]:separator[3]], color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
    plt.xlabel("Vesicle radi (px)")
    
    ## 20% DOPS, 3% PIP2
    plt.figure(dpi= 150)
    plt.title("20% DOPS, 3% PIP2 - Vesicle radi distribution")
    sns.histplot(all_radi[separator[3]:separator[4]], color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
    plt.xlabel("Vesicle radi (px)")
    
    ## 20% DOPS, 5% PIP2
    plt.figure(dpi= 150)
    plt.title("20% DOPS, 5% PIP2 - Vesicle radi distribution")
    sns.histplot(all_radi[separator[4]:separator[5]], color="dodgerblue", alpha=0.5, edgecolor='none')#, label="100% DOPC\n N$_{vesicles}$ = " + str(int(len(data))))#, **kwargs)
    plt.xlabel("Vesicle radi (px)")
    







