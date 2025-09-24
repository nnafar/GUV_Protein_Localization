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



def find_projects_info(path_to_script):
    """
    Finds the experiment and image information of all the data sets to be analyzed,
    stored in the "data" folder. The membrane channel tif file's name is used as 
    reference as it is present in all sets. 
    Note: File names should be of the format: 
          "experiment_date-experiment_name-image_name-channel", where "channel" 
          should be: 
          "C1" for the membrane channel, 
          "C2", if a single protein is present, for that protein's channel,
          "C2" for the septin channel, if both proteins are present
          "C3" for the actin channel, if both proteins are present.
         
    Input
    -----
    path_to_script : str
        The path to the script. Within the same folder subfolder "data" should
        exist and contain all the files to be analyzed. 
    
    Returns
    -------
    exp_info_all_sets : np.ndarray
        All the information of the images analyzed. Information of different sets
        are stored vertically, while horizontally:
        exp_info_all_sets[:,0] is the experiment date
        exp_info_all_sets[:,1] is the experiment name
        exp_info_all_sets[:,2] is the name of the image to be analyzed
        exp_info_all_sets[:,3] is the full representative name by combining the previous information.
        
    """
    os.chdir(path_to_script + "\\data")
    
    set_of_data  = 0
    exp_info_all_sets = np.zeros((4), dtype = str)
    
    for f in os.listdir():
        
        if f.endswith("C1.tif"):
            
            name, extension = os.path.splitext(f)
            exp_info_single_set = name.split("-")
            exp_info_single_set.pop(-1)
            exp_info_single_set.append(exp_info_single_set[0] + "-" + exp_info_single_set[1] + "-" + exp_info_single_set[2])
            
            if set_of_data == 0:
                exp_info_all_sets = np.array(exp_info_single_set)
            else:
                exp_info_all_sets = np.vstack((exp_info_all_sets, exp_info_single_set))
            
            set_of_data += 1
    
    
    os.chdir(path_to_script)
    
    return exp_info_all_sets


def read_files(path_to_script, exp_info, proteins_present):
    """
    Function that reads the image files and the vesicle detection by DisGUVery.
    
    Input
    -----
    path_to_script : str
        The path to the folder containing the data to be analyzed.
        
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
        
    proteins_present : np.ndarray
        Array of booleans indicating presence (True) or abscence (False) of: 
        proteins_present[0]: Septin 
        proteins_present[1]: Actin 
    
    Returns
    -------
    channels_data : np.ndarray
        The channels of the image to be analyzed.
        channels_data[:,:,0] always corresponds to the Membrane channel
        If only one protein is present:
        channels_data[:,:,1] corresponds to that protein's channel
        If two proteins are present:
        channels_data[:,:,1] corresponds to the Septin channel    
        channels_data[:,:,2] corresponds to the Actin channel  
    
    coordinates : np.ndarray
        The vesicles' information from the DisGUVery output.
        coordinates[:,0]: vesicle ids of all vesicles
        coordinates[:,1]: xc (px) of all vesicles
        coordinates[:,2]: yc (px) of all vesicles
        coordinates[:,3]: radi (px) of all vesicles
        
    """
    os.chdir(path_to_script + "\\data")
    
    channels_data = plt.imread(exp_info[3] + "-C1.tif")
    
    if proteins_present[0] == True:
        channels_data = np.dstack((channels_data, plt.imread(exp_info[3] + "-C2.tif")))
        
    if proteins_present[1] == True: 
        channels_data = np.dstack((channels_data, plt.imread(exp_info[3] + "-C3.tif")))
    
    # Read the CSV file - but don't worry about column names yet
    csv_file_path = exp_info[3] + "-detected_vesicles.csv"
    detected_vesicles = pd.read_csv(csv_file_path)
    
    # Print data for debugging
    print("CSV columns:", detected_vesicles.columns.tolist())
    print("First few rows:")
    print(detected_vesicles.head())
    
    # Create coordinates array 
    coordinates = np.zeros((len(detected_vesicles), 4))
    
    # IMPORTANT: Explicitly assign values based on column position, not names
    # We know the correct order should be: ID, xc, yc, radius
    
    # For vesicle ID, we'll just create sequential numbers starting from 1
    coordinates[:, 0] = np.arange(1, len(detected_vesicles) + 1)
    
    # For x, y, radius - we'll use column positions from the CSV
    # Assuming standard DisGUVery format: x is column 1, y is column 2, size/radius is column 3
    coordinates[:, 1] = detected_vesicles.iloc[:, 0].values  # X coordinate
    coordinates[:, 2] = detected_vesicles.iloc[:, 1].values  # Y coordinate  
    coordinates[:, 3] = detected_vesicles.iloc[:, 2].values  # Radius/size
    
    # Print coordinate ranges for verification
    print(f"X coordinate range: {coordinates[:,1].min()} to {coordinates[:,1].max()}")
    print(f"Y coordinate range: {coordinates[:,2].min()} to {coordinates[:,2].max()}")
    print(f"Radius range: {coordinates[:,3].min()} to {coordinates[:,3].max()}")
    
    os.chdir(path_to_script)
    
    return channels_data, coordinates

def create_output(path_to_output, exp_info, proteins_present):
    """
    Creates an empty output directory and an output file. If it already existed,
    it is overwritten.
    
    Input
    -----
    path_to_output : str
        The path to the folder to output folder.
    
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
    
    proteins_present : np.ndarray
        Array of booleans indicating presence (True) or abscence (False) of: 
        proteins_present[0]: Septin 
        proteins_present[1]: Actin
        
    """
    os.makedirs(path_to_output, exist_ok= True)
        
    protein_status = plot_format(proteins_present)
    
    if protein_status == "both_proteins":
        column_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "A Background", "S localization", "A localization", "Comment"]

    if protein_status == "only_septin":
        column_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "S localization", "Comment"]

    if protein_status == "only_actin":
        column_headers = ["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "A Background", "A localization", "Comment"]

        
    with open(path_to_output+"\\" + exp_info[3] + "-Output.csv", "w", newline='') as output_file:
        df = csv.DictWriter(output_file, delimiter=',', fieldnames = column_headers)
        df.writeheader()
        
    
def color_maps():
    """
    Creates and registers in matplotlib the colormaps that will be used for the
    different channels. If the colormaps are already register the process is skipped.  
    
    Conventions of LUTs:
    --------------------
    Membrane channel:  Cyan
    Septin channel  :  Magenta
    Actin channel   :  Yellow
    
    """
    if "cmap_cyan" not in plt.colormaps():
        colors = ["black", "cyan"]
        nodes = [0.0, 1.0]
        cmap_cyan = LinearSegmentedColormap.from_list("cmap_cyan", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_cyan)  # Updated method
        print("Colormap 'cmap_cyan' has been registered in matplotlib.")
        
    if "cmap_magenta" not in plt.colormaps():
        colors = ["black","magenta"]
        nodes = [0.0, 1.0]
        cmap_magenta = LinearSegmentedColormap.from_list("cmap_magenta", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_magenta)  # Updated method
        print("Colormap 'cmap_magenta' has been registered in matplotlib.")
    
    if "cmap_magenta_enhanced" not in plt.colormaps():
        colors = ["black","magenta", "magenta"]
        nodes = [0.0, 0.2, 1.0]
        cmap_magenta_enhanced = LinearSegmentedColormap.from_list("cmap_magenta_enhanced", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_magenta_enhanced)  # Updated method
        print("Colormap 'cmap_magenta_enhanced' has been registered in matplotlib.")
        
    if "cmap_yellow" not in plt.colormaps():
        colors = ["black","yellow"]
        nodes = [0.0, 1.0]
        cmap_yellow = LinearSegmentedColormap.from_list("cmap_yellow", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_yellow)  # Updated method
        print("Colormap 'cmap_yellow' has been registered in matplotlib.")

    
def detected_centres(to_plot, num_vesicles, ch_membrane, all_xc, all_yc, path_to_output, exp_info):
    """
    Displays the image to be analyzed with the centers of the vesicles, detected
    with DisGUVery, annotated.  
    
    Input
    -----
    to_plot : bool
        If True, the image and the centers are plotted and displayed.
        If False, the function has no output.
        
    num_vesicles : int
        The total number of detected vesicles.
    
    ch_membrane : np.ndarray
        The membrane channel.
    
    all_xc : np.ndarray
        The array containing the x coordinates of all the detected vesicles.
        
    all_yc : np.ndarray
        The array containing the y coordinates of all the detected vesicles.
    
    path_to_output : str
        The path to the folder to output folder.
    
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
    
    """
    if to_plot == True:
        plt.figure(dpi=125)
        plt.title('Total number of detected vesicles: ' + str(num_vesicles))    
        plt.axis('off')
        plt.imshow(ch_membrane, cmap="cmap_cyan")
        plt.scatter(all_xc, all_yc, color='red', s=1, marker="o")
        for i in range(num_vesicles):
            plt.annotate(i+1, (all_xc[i]+5, all_yc[i]-5), c = 'red') 
        
    
        plt.savefig(path_to_output+"\\" + exp_info[3] + "-Detected_centres.png")



def zoom_in_vesicle(channel, size_side, image_dim, ves_coordinates):
    """
    Creates a square croped view of a chosen vesicle. In case the vesicle is 
    close to the border the produced image is recentered so that the original
    image borders are not crossed. 
    
    Input
    -----
    channel : np.ndarray
        The image data corresponding to the channel we want to crop.
        
    size_side : float
        The length (unit of measure: each vesicles' radius) of the side of the 
        square zoomed-in image we want to create. 
        
    image_dim : np.ndarray
        The original image's dimensions. 
        image_dim[0] corresponds to the x direction.
        image_dim[1] corresponds to the y direction.
        
    ves_coordinates : np.ndarrya
        1-dimensional array with the coordinates of the single vesicle we want 
        to focus on.  
        ves_coordinates[0]: vesicle id 
        ves_coordinates[1]: xc (in px) 
        ves_coordinates[2]: yc (in px) 
        ves_coordinates[3]: radius (in px)
        
    Returns
    -------
    vesicle_box_channel : np.ndarray
        The 2-D array with the data of the cropped image.

    """
    xc               = ves_coordinates[1]
    yc               = ves_coordinates[2]
    radius           = ves_coordinates[3]
    
    vesicle_box_side = int(size_side * radius)
    
    move_right = 0
    move_left  = 0
    move_up    = 0
    move_down  = 0

    if xc - vesicle_box_side < 0: 
        move_right = vesicle_box_side - xc
                  
    if xc + vesicle_box_side > image_dim[0]: 
        move_left  = vesicle_box_side - (image_dim[0]-xc)   
        
    if yc - vesicle_box_side < 0: 
        move_down  = vesicle_box_side - yc           
                      
    if yc + vesicle_box_side > image_dim[1]: 
        move_up    = vesicle_box_side - (image_dim[1]-yc)   
         
    vesicle_box_channel = channel[int(yc - vesicle_box_side + move_down - move_up):int(yc + vesicle_box_side + move_down - move_up), int(xc - vesicle_box_side + move_right - move_left):int(xc + vesicle_box_side - move_left + move_right)]
    
    
    return vesicle_box_channel


def define_threshold(user_choice, channel, size_side, image_dim, ves_coordinates):    
    """
    Allows for optional manual choice of the threshold method used for the 
    background noise calculation. If no manual choice is desired, the Li method
    is chosen by default.  

    Input
    -----
    user_choice : bool
        If True: manual choice is required.
        If False: automatic choice takes place.
        
    channel : np.ndarray
        The image data corresponding to the channel we want to crop. 
        
    size_side : float
        The length (unit of measure: each vesicles' radius) of the side of the 
        square zoomed-in image we want to create. 
        
    image_dim : np.ndarray
        The original image's dimensions. 
        image_dim[0] corresponds to the x direction.
        image_dim[1] corresponds to the y direction.
        
    ves_coordinates : np.ndarrya
        1-dimensional array with the coordinates of the single vesicle we want 
        to focus on.  
        ves_coordinates[0]: vesicle id 
        ves_coordinates[1]: xc (in px) 
        ves_coordinates[2]: yc (in px) 
        ves_coordinates[3]: radius (in px)

    Returns
    -------
    threshold : float
        The threshold value to be used during background calculation. 

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

    Input
    -----
    channels_data : np.ndarray
        The channels of the image to be analyzed.
        channels_data[:,:,0] always corresponds to the Membrane channel
        If only one protein is present:
        channels_data[:,:,1] corresponds to that protein's channel
        If two proteins are present:
        channels_data[:,:,1] corresponds to the Septin channel    
        channels_data[:,:,2] corresponds to the Actin channel 
        
    ves_coordinates : np.ndarrya
        1-dimensional array with the coordinates of the single vesicle we want 
        to focus on.  
        ves_coordinates[0]: vesicle id 
        ves_coordinates[1]: xc (in px) 
        ves_coordinates[2]: yc (in px) 
        ves_coordinates[3]: radius (in px)
        
    image_dim : np.ndarray
        The original image's dimensions. 
        image_dim[0] corresponds to the x direction.
        image_dim[1] corresponds to the y direction.
        
    parameters_profiles : np.ndarray
        The 3 important parameters for the linear profile calculation:
        parameters_profiles[0] : number of linear profiles to consider for a 
                                 single vesicle
        parameters_profiles[1] : length of the linear profiles 
                                 (unit of measure: vesicle radius)
        parameters_profiles[2] : the step (in px) for sampling along the vesicle
                                 radius
        
    Returns
    -------
    intensity_profiles : np.ndarray
        The intensity linear profiles calculated.
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,0] is always based on the 
              membrane channel, element [:,:,1] is based on the single protein
              present. If both proteins are present, element [:,:,1] corresponds
              to the septin channel and element [:,:,2] corresponds to the actin
              channel.
        
    along_radius : np.ndarray
        The points along the linear profiles, starting from the center of the
        vesicle. 
    
    theta : np.ndarray
        The angles considered for the linear profiles.
        
    death_mark : bool
        If True, the vesicle's requested intensity profile is extending outside
        of the image borders and the vesicle is marked to be disregarded. 
        If False, the vesicle's position doesn't raise an issue and the analysis
        can be continued.

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
          
    
    ## Defining the image indices corresponding to the pixels along the linear profiles:
        
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
    Assistive function that fills in the space inside the vesicle contours of the 
    membrane mask. 
    Additionally fills in partially visible vesicles that are cut at the image
    borders. 
    
    Input
    -----
    mask_memb : np.ndarray
        The membrane mask created using a thresholding method.
    
    Returns
    -------
    mask_memb_fill : np.ndarray
        The filled in mask to be used for the background noise calculation. 

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
    Calculation of the background noise, focusing in the region of the vesicle
    of interest. 
    Optionally, the calculated mask is diplayed for visual inspection, next to
    the original images. A choice of proper image format based on the number of 
    proteins present is incorporated. 

    Input
    -----
    plot_mask: bool
        If True, the calculated masks are also plotted for visual inspection.
        If False, no plotting takes place.
    
    channels_data : np.ndarray
        The channels of the image to be analyzed.
        channels_data[:,:,0] always corresponds to the Membrane channel
        If only one protein is present:
        channels_data[:,:,1] corresponds to that protein's channel
        If two proteins are present:
        channels_data[:,:,1] corresponds to the Septin channel    
        channels_data[:,:,2] corresponds to the Actin channel 
    
    proteins_present : np.ndarray
        Array of booleans indicating presence (True) or abscence (False) of: 
        proteins_present[0]: Septin 
        proteins_present[1]: Actin 
    
    size_mask : float
        The length (unit of measure: each vesicles' radius) of the side of the 
        square zoomed-in image of the vesicle of interest we want to consider 
        for the background calculation.  
        
    image_dim : np.ndarray
        The original image's dimensions. 
        image_dim[0] corresponds to the x direction.
        image_dim[1] corresponds to the y direction.
        
    ves_coordinates : np.ndarrya
        1-dimensional array with the coordinates of the single vesicle we want 
        to focus on.  
        ves_coordinates[0]: vesicle id 
        ves_coordinates[1]: xc (in px) 
        ves_coordinates[2]: yc (in px) 
        ves_coordinates[3]: radius (in px)
        
    threshold : float
        The threshold value to be used for the creation of the mask. 
    
    path_to_output : str
        The path to the output folder. 
        
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
        
    Returns
    -------
    background: float
        The background noise intensity estimated. Data related to different 
        channels are stored along the 3rd dimension of the array:
        background[:,0] always corresponds to the background of the membrane channel.
        If only one protein is present:
        background[:,1] corresponds to the background of that protein's channel.
        If two proteins are present:
        background[:,1] corresponds to the background of the septin channel.    
        background[:,2] corresponds to the background of the actin channel.
    
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
    channels present in the analyzed image. 
    
    Input
    -----
    proteins_present : np.ndarray
        Array of booleans indicating presence (True) or abscence (False) of: 
        proteins_present[0]: Septin 
        proteins_present[1]: Actin 
        
    Returns
    -------
    protein_status: str
        A string indicating how many and which protein are present.
        Three options possible: only_septin, only_actin, both_proteins. 
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
    Displays the background noise calculated and the produced mask for visual 
    inspection, next to the original images.
    Format is appropiate only when both proteins are present.  
    
    Input
    -----
    vesicle_id : int
        The vesicle's identity number.
    
    vesicle_box : np.ndarray
        The square cropped view of a chosen vesicle. 
        vesicle_box[:,:,0] corresponds to the membrane channel.
        vesicle_box[:,:,1] corresponds to the septin channel.
        vesicle_box[:,:,2] corresponds to the actin channel.
    
    mask_memb_fill : np.ndarray
        The filled in mask to be used for the background noise calculation.
    
    background: float
        The background noise intensity estimated. Data related to different 
        channels are stored along the 3rd dimension of the array:
        background[:,0] always corresponds to the background of the membrane channel.
        background[:,1] corresponds to the background of the septin channel.    
        background[:,2] corresponds to the background of the actin channel.
    
    path_to_output : str
        The path to the output folder. 
        
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
        
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
    
    fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Channels_and_Mask.png")

    
def show_mask_only_sept(vesicle_id, vesicle_box, mask_memb_fill, background, path_to_output, exp_info):
   """
   Displays the background noise calculated and the produced mask for visual 
   inspection, next to the original images.
   Format is appropiate when only septin is present. 
   
   Input
   -----
   vesicle_id : int
       The vesicle's identity number.
   
   vesicle_box : np.ndarray
       The square cropped view of a chosen vesicle. 
       vesicle_box[:,:,0] corresponds to the membrane channel.
       vesicle_box[:,:,1] corresponds to the septin channel.
       vesicle_box[:,:,2] corresponds to the actin channel.
   
   mask_memb_fill : np.ndarray
       The filled in mask to be used for the background noise calculation.
   
   background: float
       The background noise intensity estimated. Data related to different 
       channels are stored along the 3rd dimension of the array:
       background[:,0] always corresponds to the background of the membrane channel.
       background[:,1] corresponds to the background of the septin channel.    
    
   path_to_output : str
       The path to the output folder. 
        
   exp_info : list
       Contains information of the file's name in order:
       exp_info[0]: the experiment date
       exp_info[1]: the experiment name
       exp_info[2]: the name of the image to be analyzed
       exp_info[3]: full representative name by combining the previous information
        
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

   fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Channels_and_Mask.png")


def show_mask_only_actin(vesicle_id, vesicle_box, mask_memb_fill, background, path_to_output, exp_info):
   """
   Displays the background noise calculated and the produced mask for visual 
   inspection, next to the original images.
   Format is appropiate when only actin is present. 
   
   Input
   -----
   vesicle_id : int
       The vesicle's identity number.
   
   vesicle_box : np.ndarray
       The square cropped view of a chosen vesicle. 
       vesicle_box[:,:,0] corresponds to the membrane channel.
       vesicle_box[:,:,1] corresponds to the septin channel.
       vesicle_box[:,:,2] corresponds to the actin channel.
   
   mask_memb_fill : np.ndarray
       The filled in mask to be used for the background noise calculation.
   
   background: float
       The background noise intensity estimated. Data related to different 
       channels are stored along the 3rd dimension of the array:
       background[:,0] always corresponds to the background of the membrane channel.
       background[:,1] corresponds to the background of the actin channel.    
    
   path_to_output : str
       The path to the output folder. 
     
   exp_info : list
       Contains information of the file's name in order:
       exp_info[0]: the experiment date
       exp_info[1]: the experiment name
       exp_info[2]: the name of the image to be analyzed
       exp_info[3]: full representative name by combining the previous information     
   
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
  
   fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Channels_and_Mask.png")


def background_correction(num_channels, intensity_profiles, background):
    """ 
    Background correction of the intesity linear profiles. 
    
    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed.
        
    intensity_profiles : np.ndarray
        The intensity linear profiles calculated.
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,0] is always based on the 
              membrane channel, element [:,:,1] is based on the single protein
              present. If both proteins are present, element [:,:,1] corresponds
              to the septin channel and element [:,:,2] corresponds to the actin
              channel.
              
    background : np.ndarray
        The background noise estimated for the different channels
        Data related to different 
        channels are stored along the 3rd dimension of the array:
        background[:,0] always corresponds to the background of the membrane channel.
        If only one protein is present:
        background[:,1] corresponds to the background of that protein's channel.
        If two proteins are present:
        background[:,1] corresponds to the background of the septin channel.    
        background[:,2] corresponds to the background of the actin channel.
    
    Returns
    -------
    intensity_profiles_corrected : np.ndarray
        The intensity linear profiles after the background noise correction.
        Note: Data is stored as in the intensity_profiles array. 
              
    """
    intensity_profiles_corrected = np.ones_like(intensity_profiles)
    
    for i in range(num_channels):
        intensity_profiles_corrected[:,:,i] = np.clip(intensity_profiles[:,:,i] - background[:,i], 0, None) # setting negative values occuring after subtraction to 0

    return intensity_profiles_corrected


def radial_profile(num_channels, intensity_profiles):
    """
    Calculation of a collective radial profile for a given vesicle from multiple 
    equally-distances linear profiles along the vesicles radius. 
    
    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed. 
        
    intensity_profiles : np.ndarray
        The intensity linear profiles calculated. Backgrounds corrections should
        take place beforehand. 
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,0] is always based on the 
              membrane channel, element [:,:,1] is based on the single protein
              present. If both proteins are present, element [:,:,1] corresponds
              to the septin channel and element [:,:,2] corresponds to the actin
              channel.
    
    Returns
    -------
    radial_profiles : np.ndarray
        The collective radial intensity profile of the vesicle under study. 
    
    """
    radial_profiles  = np.ones_like(intensity_profiles[:,0,:])
    
    for i in range(num_channels):
        radial_profiles[:,i]  = np.average(intensity_profiles[:,:,i], axis = 1)
         
    return radial_profiles


def membrane_detection(radial_profile_memb, radius):
    """
    Detection of the membrane inner and outer borders.
    Convention used: width at half maximum
    
    Input
    -----
    radial_profile_memb : np.ndarray
        The 1D array of the membrane radial intensity profile. 
    
    radius: float
        The radius of the vesicle detected by DisGUVery. 
        
    Returns
    -------
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
    
    comment : list
        List of strings that correspond to comments on issues encountered.
        
    death_mark : bool
        If True, no peak was detected and the vesicle is marked to be disregarded. 
        If False, peak/s was/were detected and the analysis can continued.
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
    
    Input
    -----
    angular_profiles : np.ndarray
        The angular profile along the membrane contour for a given vesicle.
        
    radial_profiles : np.ndarray
        The collective radial intensity profile of the vesicle under study.
        
    radius : float
        The radius of the vesicle under study.
    
    Returns
    -------
    comments : list
        List of quality flags for manual review
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
    Calculation of the angular profile along the membrane contour for a given 
    vesicle.
    Current convention: Averaging over the detected membrane area for each angle
                        considered during the calcualtion of the linear profiles. 
                        
    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed.

    intensity_profiles : np.ndarray
        The intensity linear profiles calculated. Backgrounds corrections should
        take place beforehand. 
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,0] is always based on the 
              membrane channel, element [:,:,1] is based on the single protein
              present. If both proteins are present, element [:,:,1] corresponds
              to the septin channel and element [:,:,2] corresponds to the actin
              channel.
              
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
    
    pixels_to_remove : int
        The number of pixels at the center to be removed to avoid the inherent 
        pixel multicounting.
        
    Returns
    -------
    angular_profiles : np.ndarray
        The angular profile along the membrane contour for a given vesicle.
        The profiles corresponding to the different channels are stored along 
        the 3rd dimension. 
        angular_profiles[:,0] corresponds to the membrane channel.
        If only one protein is present:
        angular_profiles[:,1] corresponds to that protein channel. 
        If both proteins are peresent:
        angular_profiles[:,1] corresponds to the septin channel. 
        angular_profiles[:,2] corresponds to the actin channel.
    
    """
    # intensity_profiles = intensity_profiles[:, pixels_to_remove:, :]
    angular_profiles = np.ones_like(intensity_profiles[0,:,:])
    
    for i in range(num_channels):
        angular_profiles[:,i] = np.average(intensity_profiles[pixels_to_remove+index_border_in:pixels_to_remove+index_border_out,:,i], axis=0)
    
    # print("angular_profiles", np.average(angular_profiles[:,1]))
    return angular_profiles


def localization(num_channels, angular_profiles, radial_profiles, index_border_in, index_border_out, radius, size_central_area, pixels_to_remove):
    """
    Quantification of the protein localization on the membrane.
    Convention used: Difference between the median signal within the detected
                     membrane width the average signal at a central area of the
                     vesicle, normalized by the median signal within the detected
                     membrane width. Negative values are clipped to 0.

   Recommended values for classification:    0  < L < 1/6  : No localization
                                            1/6 < L <= 1   : Localization              
    
    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed.
    
    angular_profiles : np.ndarray
        The collective angular intensity profile along the detected membrane of
        the vesicle under study. 
        
    radial_profiles : np.ndarray
        The collective radial intensity profile of the vesicle under study. 
        
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
    
    radius : float
        The radius of the vesicle under study.
        
    size_central_area : float
        The central area radius (unit of measure: each vesicles' radius) to be 
        considered.
    pixels_to_remove : int
        The number of pixels at the center to be removed to avoid the inherent 
        pixel multicounting.
    
    Returns
    -------
    localization : np.ndarray
        The quantified localization of the protein(s) at the membrane.
        If only one protein is present, then localization has a single element
        corresponding to that protein. 
        If both proteins are present, localization[0] corresponds to the septin
        channel and localization[1] corresponds to the actin channel. 
        
    comment : list
        List of strings that correspond to comments on issues encountered/
    
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
    Optional plotting of the calculated intensity radial and angular profiles 
    next to the vesicle channels for easier visual inspection.
    A choice of proper image format based on the number of proteins present is 
    incorporated. 
    
    Input
    -----
    plot_intensity_profiles : bool
        If True, the intesnity profiles are displayed. 
        If False, the function is skipped.
    
    proteins_present : np.ndarray
        Array of booleans indicating presence (True) or abscence (False) of: 
        proteins_present[0]: Septin 
        proteins_present[1]: Actin  
    
    parameters_sizes : np.ndarray
        The parameters related to the size of cropped mages used in the code and 
        the conention for the border fo the central vesicle area considered. 
        parameters_sizes[0] relates to the size of the considered mask calculation.
        parameters_sizes[1] relates to the size of the image used for visual inspection.
        parameters_sizes[2] relates to the size of the considered central vesicle area. 

    channels_data : np.ndarray
        The channels of the image to be analyzed.
        channels_data[:,:,0] always corresponds to the Membrane channel
        If only one protein is present:
        channels_data[:,:,1] corresponds to that protein's channel
        If two proteins are present:
        channels_data[:,:,1] corresponds to the Septin channel    
        channels_data[:,:,2] corresponds to the Actin channel 
    
    ves_coordinates : np.ndarrya
        1-dimensional array with the coordinates of the single vesicle we want 
        to focus on.  
        ves_coordinates[0]: vesicle id 
        ves_coordinates[1]: xc (in px) 
        ves_coordinates[2]: yc (in px) 
        ves_coordinates[3]: radius (in px)
    
    image_dim : np.ndarray
        The original image's dimensions. 
        image_dim[0] corresponds to the x direction.
        image_dim[1] corresponds to the y direction.
    
    along_radius : np.ndarray
        The points along the linear profiles, starting from the center of the
        vesicle. 
    
    theta : np.ndarray
        The angles considered for the linear profiles.
    
    radial_profiles : np.ndarray
        The collective radial intensity profile of the vesicle under study. 
    
    angular_profiles : np.ndarray
        The angular profile along the membrane contour for a given vesicle.
        The profiles corresponding to the different channels are stored along 
        the 3rd dimension. 
        angular_profiles[:,0] corresponds to the membrane channel.
        If only one protein is present:
        angular_profiles[:,1] corresponds to that protein channel. 
        If both proteins are peresent:
        angular_profiles[:,1] corresponds to the septin channel. 
        angular_profiles[:,2] corresponds to the actin channel.
    
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
    
    background : np.ndarray
        The background noise estimated for the different channels
        Data related to different 
        channels are stored along the 3rd dimension of the array:
        background[:,0] always corresponds to the background of the membrane channel.
        If only one protein is present:
        background[:,1] corresponds to the background of that protein's channel.
        If two proteins are present:
        background[:,1] corresponds to the background of the septin channel.    
        background[:,2] corresponds to the background of the actin channel.
    
    localization : np.ndarray
        The quantified localization of the protein(s) at the membrane.
        If only one protein is present, then localization has a single element
        corresponding to that protein. 
        If both proteins are present, localization[0] corresponds to the septin
        channel and localization[1] corresponds to the actin channel. 
    
    path_to_output : str
        The path to the output folder. 
        
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
        
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
    Plotting of the intensity radial and angular profiles. 
    Appropriate format only when both proteins are present.
    
    Input
    -----
    vesicle_id : int
        The vesicle's identity number.
        
    radius : float
        The vesicle's radius detected by DisGUVery.   
        
    size_central_area : float
        The central area radius (unit of measure: each vesicles' radius) considered during localization quantification
    
    along_radius : np.ndarray
        The points along the linear profiles, starting from the center of the
        vesicle. 
    
    theta : np.ndarray
        The angles considered for the linear profiles.
    
    radial_profiles : np.ndarray
        The collective radial intensity profile of the vesicle under study. 
    
    angular_profiles : np.ndarray
        The angular profile along the membrane contour for a given vesicle.
        The profiles corresponding to the different channels are stored along 
        the 3rd dimension. 
        angular_profiles[:,0] corresponds to the membrane channel.
        angular_profiles[:,1] corresponds to the septin channel. 
        angular_profiles[:,2] corresponds to the actin channel.
    
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
    
    image_box : np.ndarray
        The cropped image of the vesicle to be displayed. 
        image_box[:,:,0] corresponds to the membrane channel.
        image_box[:,:,1] corresponds to the septin channel. 
        image_box[:,:,2] corresponds to the septin channel. 
        
    background : np.ndarray
        The background noise estimated for the different channels
        Data related to different 
        channels are stored along the 3rd dimension of the array:
        background[:,0] corresponds to the background of the membrane channel.
        background[:,1] corresponds to the background of the septin channel.    
        background[:,2] corresponds to the background of the actin channel.
    
    localization : np.ndarray
        The quantified localization of the proteins at the membrane.
        localization[0] corresponds to the septin channel.
        localization[1] corresponds to the actin channel. 
    
    path_to_output: str
        The path to the output folder.
    
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
    
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
    
    fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Radial.png")
    
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
    
    fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Angular.png")

    
def plot_int_prof_only_septin(vesicle_id, radius, size_central_area, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info):
    """
    Plotting of the intensity radial and angular profiles. 
    Appropriate format when only septin is present.
    
    Input
    -----
    vesicle_id : int
        The vesicle's identity number.
        
    radius : float
        The vesicle's radius detected by DisGUVery.   
    
    size_central_area : float
        The central area radius (unit of measure: each vesicles' radius) considered during localization quantification
    
    along_radius : np.ndarray
        The points along the linear profiles, starting from the center of the
        vesicle. 
    
    theta : np.ndarray
        The angles considered for the linear profiles.
    
    radial_profiles : np.ndarray
        The collective radial intensity profile of the vesicle under study. 
    
    angular_profiles : np.ndarray
        The angular profile along the membrane contour for a given vesicle.
        The profiles corresponding to the different channels are stored along 
        the 3rd dimension. 
        angular_profiles[:,0] corresponds to the membrane channel.
        angular_profiles[:,1] corresponds to the septin channel. 
    
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
    
    image_box : np.ndarray
        The cropped image of the vesicle to be displayed. 
        image_box[:,:,0] corresponds to the membrane channel.
        image_box[:,:,1] corresponds to the septin channel. 
        
    background : np.ndarray
        The background noise estimated for the different channels
        Data related to different 
        channels are stored along the 3rd dimension of the array:
        background[:,0] corresponds to the background of the membrane channel.
        background[:,1] corresponds to the background of the septin channel.    
    
    localization : np.ndarray
        The quantified localization of the protein at the membrane.
        localization[0] corresponds to the septin channel.
    
    path_to_output: str
        The path to the output folder.
    
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
    
    """
    ## Intensity radial profile:
        
    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [6, 2], dpi=125)
    axes["A"].set_title('Intensity radial profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(along_radius[2:], radial_profiles[2:,0]/np.max(radial_profiles[2:,0]), c = 'cyan', label = 'Membrane channel')
    axes["A"].plot(along_radius[2:], radial_profiles[2:,1]/np.max(radial_profiles[2:,1]), c = 'magenta', label = 'Septin channel')
    # axes["A"].axvline(x = along_radius[index_border_in], c = 'red', label = 'border in', ls="--")
    # axes["A"].axvline(x = along_radius[index_border_out], c = 'orange', label = 'border out', ls="--")
    # axes["A"].axvline(x = radius, c = 'black', label = 'detected radius', ls=":")
    # axes["A"].axvline(x = size_central_area*radius, c ='green', label = 'central area border')
    axes["A"].set(xlabel = 'radius (px)', ylabel = 'normalized intensity (a.u.)')
    # axes["A"].legend(loc="upper left", fontsize=10)
    axes["B"].set_title("Membrane channel", fontsize = 10, pad =-1)
    axes["B"].imshow(image_box[:,:,0], cmap = "cmap_cyan")
    axes["B"].set_axis_off()
    axes["C"].set_title("Septin channel", fontsize = 10, pad =-1)
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced")
    axes["C"].set_axis_off()
    # axes["D"].text(-0.2,0.7,"Background M:  " + str(round(background[0,0],4)), size = 10)
    # axes["D"].text(-0.2,0.5,"Background S:   " + str(round(background[0,1],4)), size = 10)
    # axes["D"].text(-0.2,0.0,"Localization S :  " + str(round(localization[0],3)), size = 10)
    # axes["D"].set_axis_off()
    
    fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Radial.png")
    
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
    
    fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Angular.png")
    
    
def plot_int_prof_only_actin(vesicle_id, radius, size_central_area, along_radius, theta, radial_profiles, angular_profiles, index_border_in, index_border_out, image_box, background, localization, path_to_output, exp_info):
    """
    Plotting of the intensity radial and angular profiles. 
    Appropriate format when only septin is present.
    
    Input
    -----
    vesicle_id : int
        The vesicle's identity number.
        
    radius : float
        The vesicle's radius detected by DisGUVery.   
    
    size_central_area : float
        The central area radius (unit of measure: each vesicles' radius) considered during localization quantification
    
    along_radius : np.ndarray
        The points along the linear profiles, starting from the center of the
        vesicle. 
    
    theta : np.ndarray
        The angles considered for the linear profiles.
    
    radial_profiles : np.ndarray
        The collective radial intensity profile of the vesicle under study. 
    
    angular_profiles : np.ndarray
        The angular profile along the membrane contour for a given vesicle.
        The profiles corresponding to the different channels are stored along 
        the 3rd dimension. 
        angular_profiles[:,0] corresponds to the membrane channel.
        angular_profiles[:,1] corresponds to the actin channel. 
    
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
    
    image_box : np.ndarray
        The cropped image of the vesicle to be displayed. 
        image_box[:,:,0] corresponds to the membrane channel.
        image_box[:,:,1] corresponds to the actin channel. 
        
    background : np.ndarray
        The background noise estimated for the different channels
        Data related to different 
        channels are stored along the 3rd dimension of the array:
        background[:,0] corresponds to the background of the membrane channel.
        background[:,1] corresponds to the background of the actin channel.    
    
    localization : np.ndarray
        The quantified localization of the protein at the membrane.
        localization[0] corresponds to the actin channel.
        
    path_to_output: str
        The path to the output folder.   
    
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
    
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
    
    fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Radial.png")
    
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

    fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Angular.png")


def edit_output(path_to_output, exp_info, ves_coordinates, background, localization, comment):
    """
    Adds all relevant information for a single vesicle in a row of the previously
    created output file.

    Parameters
    ----------
    path_to_output : str
        The path to the folder to output folder.
    
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
    
    ves_coordinates : np.ndarrya
        1-dimensional array with the coordinates of the single vesicle we want 
        to focus on.  
        ves_coordinates[0]: vesicle id 
        ves_coordinates[1]: xc (in px) 
        ves_coordinates[2]: yc (in px) 
        ves_coordinates[3]: radius (in px)
        
     background : np.ndarray
         The background noise estimated for the different channels
         Data related to different 
         channels are stored along the 3rd dimension of the array:
         background[:,0] always corresponds to the background of the membrane channel.
         If only one protein is present:
         background[:,1] corresponds to the background of that protein's channel.
         If two proteins are present:
         background[:,1] corresponds to the background of the septin channel.    
         background[:,2] corresponds to the background of the actin channel.
     
     localization : np.ndarray
         The quantified localization of the protein(s) at the membrane.
         If only one protein is present, then localization has a single element
         corresponding to that protein. 
         If both proteins are present, localization[0] corresponds to the septin
         channel and localization[1] corresponds to the actin channel. 
     
    comment : list
         A list of strings with relevant comments explaining (possible) issues 
         related to the specific vesicle. 

    """
    list_exp_info        = list(exp_info[:-1])
    list_ves_coordinates = list(ves_coordinates)
    list_background      = list(background[0,:])
    list_localization    = list(localization)


    vesicle_row = list_exp_info +  list_ves_coordinates + list_background + list_localization + comment 
    
    with open(path_to_output+"\\" + exp_info[3] + "-Output.csv", "a", newline='') as output_file:
               
        writer = csv.writer(output_file)
        writer.writerow(vesicle_row)
    