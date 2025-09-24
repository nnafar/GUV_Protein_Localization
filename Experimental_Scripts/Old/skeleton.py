# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:33:44 2022

@author: kkourkoulou


Script to calculate and analyze the intensity profiles of GUVs detected with 
DisGUVery. 

Collection of functions used in main.py

"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statistics

import pandas as pd                                                            # for accessing the csv file 
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap                          # for creating colormaps matching our conventions 
from matplotlib import colormaps as cm_list

    

from skimage.filters import try_all_threshold                                  # for background noise estimation
from skimage.filters import threshold_li, threshold_otsu, threshold_yen, \
                            threshold_isodata, threshold_mean, threshold_minimum, threshold_triangle
                                  
from scipy import ndimage as nd                                                # for filling mask contours
from scipy.signal import find_peaks, peak_widths                               # for membrane detection



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
        column_headers = ["Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "A Background", "S localization", "A localization", "Comment"]
        #["Date","Name", "Image", "Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "A Background", "S localization", "A localization", "Comment"]

    if protein_status == "only_septin":
        column_headers = ["Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "A Background", "S localization", "A localization", "Comment"]

    if protein_status == "only_actin":
        column_headers = ["Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "A Background", "S localization", "A localization", "Comment"]

    if protein_status == "none":
        column_headers = ["Vesicle id", "xc", "yc", "Radius", "M Background", "S Background", "A Background", "S localization", "A localization", "Comment"]
        
        
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
    colormaps = {}
    
    # Create and register cyan colormap
    if "cmap_cyan" not in plt.colormaps():
        colors = [(0, 0, 0), (0, 1, 1)]  # black to cyan
        nodes = [0.0, 1.0]
        cmap_cyan = LinearSegmentedColormap.from_list("cmap_cyan", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_cyan)
        print("Colormap 'cmap_cyan' has been registered in matplotlib.")
    
    # Create and register magenta colormap
    if "cmap_magenta" not in plt.colormaps():
        colors = [(0, 0, 0), (1, 0, 1)]  # black to magenta
        nodes = [0.0, 1.0]
        cmap_magenta = LinearSegmentedColormap.from_list("cmap_magenta", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_magenta)
        print("Colormap 'cmap_magenta' has been registered in matplotlib.")
    
    # Create and register yellow colormap
    if "cmap_yellow" not in plt.colormaps():
        colors = [(0, 0, 0), (1, 1, 0)]  # black to yellow
        nodes = [0.0, 1.0]
        cmap_yellow = LinearSegmentedColormap.from_list("cmap_yellow", list(zip(nodes, colors)))
        plt.colormaps.register(cmap=cmap_yellow)
        print("Colormap 'cmap_yellow' has been registered in matplotlib.")
    
    # Add colormaps to the dictionary
    colormaps["cmap_cyan"] = plt.colormaps["cmap_cyan"]
    colormaps["cmap_magenta"] = plt.colormaps["cmap_magenta"]
    colormaps["cmap_yellow"] = plt.colormaps["cmap_yellow"]
    
    return colormaps   
    
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
        plt.figure(dpi=150)
        plt.title('Total number of detected vesicles: ' + str(num_vesicles))    
        plt.axis('off')
        
        # Display the membrane channel image with proper orientation
        plt.imshow(ch_membrane, cmap="cmap_cyan", origin='upper')
        
        # Print coordinate ranges for debugging
        print(f"X-coordinate range: {min(all_xc)} to {max(all_xc)}")
        print(f"Y-coordinate range: {min(all_yc)} to {max(all_yc)}")
        print(f"Image dimensions: {ch_membrane.shape}")
        
        # Plot the detected vesicle centers
        plt.scatter(all_xc, all_yc, color='red', s=1, marker="o")
        
        # Add annotations with vesicle numbers
        for i in range(num_vesicles):
            plt.annotate(i+1, (all_xc[i]+5, all_yc[i]-5), c='red')
        
        plt.savefig(path_to_output+"\\" + exp_info[3] + "-Detected_centres.png")
        plt.close()  # Close the figure to free memory



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
    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]
    
    # Check for NaN values
    if np.isnan(xc) or np.isnan(yc) or np.isnan(radius):
        print("Warning: NaN values in coordinates. Using defaults.")
        # Use sensible defaults
        xc = image_dim[0] // 2 if np.isnan(xc) else xc
        yc = image_dim[1] // 2 if np.isnan(yc) else yc
        radius = 20.0 if np.isnan(radius) else radius
    
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
    
    # Ensure all parameters are integers for array indexing
    xc_int = int(xc)
    yc_int = int(yc)
    
    # Calculate box boundaries with safety checks
    y_start = max(0, int(yc_int - vesicle_box_side + move_down - move_up))
    y_end = min(channel.shape[0], int(yc_int + vesicle_box_side + move_down - move_up))
    x_start = max(0, int(xc_int - vesicle_box_side + move_right - move_left))
    x_end = min(channel.shape[1], int(xc_int + vesicle_box_side - move_left + move_right))
    
    vesicle_box_channel = channel[y_start:y_end, x_start:x_end]
    
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
        else:
            protein_status = "none"

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
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(13,4), dpi=150)
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
   fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(13,5), dpi=150)
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
   fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(13,5), dpi=150)
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
    # radial_profiles  = np.ones_like(intensity_profiles[:,0,:])
    
    # for i in range(num_channels):
    #     radial_profiles[:,i]  = np.average(intensity_profiles[:,:,i], axis = 1)
    
    # Create a safe placeholder for the result
    profile_length = intensity_profiles.shape[0]
    radial_profiles = np.zeros((profile_length, num_channels))
    
    # Check if the second dimension is empty
    if intensity_profiles.shape[1] == 0:
        print("Warning: intensity_profiles has zero length in second dimension!")
        return radial_profiles
    
    for i in range(num_channels):
        try:
            # Use axis=1 to average across angles for each radius
            radial_profiles[:,i] = np.mean(intensity_profiles[:,:,i], axis=1)
        except Exception as e:
            print(f"Error calculating average for channel {i}: {e}")
            print(f"intensity_profiles shape: {intensity_profiles.shape}")
            # Provide a fallback value if averaging fails
            radial_profiles[:,i] = intensity_profiles[:,0,i] if intensity_profiles.shape[1] > 0 else np.zeros(profile_length)
         
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
        List of strings that correspond to comments on issues encountered/
        
    death_mark : bool
        If True, no peak was detected and the vesicle is marked to be disregarded. 
        If False, peak/s was/were detected and the analysis can continued.
    """
    comment = []
    death_mark = False
    peaks, _ = find_peaks(radial_profile_memb, height=10, distance=5, prominence=1)
    
    if not np.any(peaks) :
        index_border_in  = 0
        index_border_out = 0 
        comment          = ["no_memb_peak"]
        death_mark       = True
        return index_border_in, index_border_out, comment, death_mark
        
    width_half_max = peak_widths(radial_profile_memb, peaks, rel_height=0.5)
    
    chosen_peak = -1
    
    if len(peaks) > 1:
        if peaks[chosen_peak] > radius and radial_profile_memb[peaks[chosen_peak]] < radial_profile_memb[peaks[chosen_peak-1]]:
            chosen_peak = chosen_peak - 1
            
            if len(peaks) > 2:
                if peaks[chosen_peak] > radius and radial_profile_memb[chosen_peak] < radial_profile_memb[chosen_peak - 1]:
                    chosen_peak = chosen_peak - 1
                    
        elif len(peaks)>2:
            if peaks[chosen_peak] > radius and radial_profile_memb[peaks[chosen_peak]] < radial_profile_memb[peaks[chosen_peak-2]]:
                chosen_peak = chosen_peak - 2
        
        if len(peaks[:chosen_peak]) != 0:
            comment = comment + ["confetti"] 
    
    # if radial_profile_memb[peaks[chosen_peak]] < max(radial_profile_memb[:peaks[chosen_peak]]):
    #     comment = comment + ["confetti"] 
   
    index_border_in = int(np.rint(width_half_max[2][chosen_peak]))
    index_border_out = int(np.rint(width_half_max[3][chosen_peak]))
    
    if index_border_out - index_border_in > radius/4:
        comment = comment + ["wide_peak"]
    
    if np.average(radial_profile_memb[:index_border_in]) >= 0.3 *radial_profile_memb[peaks[chosen_peak]]:
        comment = comment + ["lipids_inside"]
    
    if np.average(radial_profile_memb[index_border_out:]) >= 0.4 *radial_profile_memb[peaks[chosen_peak]]:
        comment = comment + ["clusters_outside"]
    
    return index_border_in, index_border_out, comment, death_mark


def localization(num_channels, radial_profiles, index_border_in, index_border_out, radius, size_central_area, pixels_to_remove):
    """
    Quantification of the protein localization on the membrane.
    Convention used: Ratio of the average protein intensity within the detected
                     membrane borders over the average protein intencity within 
                     a user defined central radius. 
    
    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed.
    
    radial_profile : np.ndarray
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
    
    """
    num_proteins   = num_channels - 1 
    index_centre   = int(size_central_area * (radius-pixels_to_remove)) 
 
    protein_bound  = np.zeros(num_proteins)
    protein_centre = np.zeros(num_proteins)
    localization   = np.zeros(num_proteins) 
    
    for i in range(num_proteins):
        protein_bound[i]  = np.average(radial_profiles[index_border_in:index_border_out,i+1])
        protein_centre[i] = np.average(radial_profiles[0:index_centre,i+1])  
        print("bound", protein_bound)
        print("centre", protein_centre)
        localization[i]   = protein_bound[i]/protein_centre[i]
    
    return localization




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


def localization_corrected(num_channels, angular_profiles, radial_profiles, index_border_in, index_border_out, radius, size_central_area, pixels_to_remove):
    """
    Quantification of the protein localization on the membrane.
    Convention used: Ratio of the average protein intensity within the detected
                     membrane borders, trying to consider only points corresponding 
                     to a uniformal septin signal and avoiding clusters, over the
                     average protein intencity within a user defined central radius. 
                     To avoid the clusters a threshold is considered using the 
                     mean of all the data to block out the points corresponding to 
                     peaks due to clusters. Vesicles with a uniform ring are treated
                     differently than vesicles with clusters, identified based on 
                     the rsd value. 
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
    
    """
    comment = []
    
    num_proteins   = num_channels - 1 
    index_centre   = int(size_central_area * (radius-pixels_to_remove)) 
 
    protein_bound  = np.zeros(num_proteins)
    protein_centre = np.zeros(num_proteins)
    localization   = np.zeros(num_proteins) 

    for i in range(num_proteins):
        ## Calculation or relative standard deviation of angular protein signal:
        rsd = np.std(angular_profiles[:,i+1])/np.mean(angular_profiles[:,i+1])
        
        ## Mean value of protein channel, regardless of the presence of clusters:    
        mean = np.mean(angular_profiles[:,i+1])
        
        if mean == 0:
            corrected_average = 0
        else:
            corrected_average = (np.average(angular_profiles[:,i+1], weights=angular_profiles[:,i+1]< mean + (mean-np.min(angular_profiles[:,i+1]))))
        
        if rsd > 0.8:
            comment = ["high_rsd"]
            
            ## Calculation of a moving average to smooth the the angular profile:
            window_size = 30
            j = 0
            moving_averages = []
            
            while j < len(angular_profiles[:,1]) - window_size + 1:
                window_average = round(np.sum(angular_profiles[j:j+window_size,1]) / window_size, 2)
                moving_averages.append(window_average)
                j += 1
            
            corrected_average = (np.average(angular_profiles[:,i+1], weights=angular_profiles[:,i+1]< mean + (mean-np.min(moving_averages))))
        
        ## Mean value of protein channel, avoiding the segments were the signal is more 
        ## than two times higher than the original mean: 
        protein_bound[i]  = corrected_average
        protein_centre[i] = np.average(radial_profiles[0:index_centre,i+1])  
        
        localization[i]   = protein_bound[i]/protein_centre[i]
    
    return localization, comment


def localization_median(num_channels, angular_profiles, radial_profiles, index_border_in, index_border_out, radius, size_central_area, pixels_to_remove):
    """
    Quantification of the protein localization on the membrane.
    Convention used: Ratio of the median of the protein intensity within the 
                     detected membrane borders, to avoid overestimation due to 
                     clusters, over the average protein intencity within a user
                     defined central radius. 
    
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
    
    """
    comment = []
    
    num_proteins   = num_channels - 1 
    index_centre   = int(size_central_area * (radius-pixels_to_remove)) 
 
    protein_bound  = np.zeros(num_proteins)
    protein_centre = np.zeros(num_proteins)
    localization   = np.zeros(num_proteins) 

    for i in range(num_proteins):
        ## Calculation or relative standard deviation of angular protein signal:
        rsd = np.std(angular_profiles[:,i+1])/np.mean(angular_profiles[:,i+1])
        
        ## Median value of protein channel:
        median = np.median(angular_profiles[:,i+1])
        
        ## Mean value of protein channel, regardless of the presence of clusters:    
        mean = np.mean(angular_profiles[:,i+1])
        
        ## Corrected average: The median is considered as the representative value:
        corrected_average = median
            
        ## Mean value of protein channel, avoiding the segments were the signal is more 
        ## than two times higher than the original mean: 
        protein_bound[i]  = corrected_average
        protein_centre[i] = np.average(radial_profiles[0:index_centre,i+1])  
        
        localization[i]   = protein_bound[i]/protein_centre[i]
    
    return localization, comment


def angular_variance_assistant(localization, angular_profiles, theta, channels_data, parameters_sizes, ves_coordinates, image_dim):
    """
    Assistive funtion for correct protein-membrane co-localization calculation
    disregarding clusters present on the membrane circumference. 
    """       
    comment = []
    
    num_channels     = len(channels_data[0,0,:])
    size_box = int(parameters_sizes[1] * ves_coordinates[3])
    vesicle_id = int(ves_coordinates[0])
    
    ## Crating zoomed-in image of the vesicle:
    image_box = np.zeros((2*size_box, 2*size_box, num_channels))

    for i in range(num_channels):
        image_box[:,:,i] = zoom_in_vesicle(channels_data[:,:,i], parameters_sizes[1], image_dim, ves_coordinates)
        
    ## Mean value of septin channel, regardless of the presence of clusters:    
    mean = np.mean(angular_profiles[:,1])
    median = np.median(angular_profiles[:,1])
    ## Mean value of septin channel, avoiding the segments were the signal is more 
    ## than two times higher than the original mean: 
    
    corrected_average = (np.average(angular_profiles[:,1], weights=angular_profiles[:,1]<mean + (mean-np.min(angular_profiles[:,1]))))
    
    ## ALTERNATIVE: If the relative standard deviation is more than 80%, peaks
    ## present in the data are detected to determine the positions of the clusters. 
    ## 
    
    ## Calculation or relative standard deviation to identify whether the vesicle
    ## has clusters on the membrane: 
    
    rsd = np.std(angular_profiles[:,1])/np.mean(angular_profiles[:,1])
    
    if rsd > 0.8:
        comment = ["high_rsd"]
    
        ## Peak detection:
        peaks, _ = find_peaks(angular_profiles[:,1], height=2*np.mean(angular_profiles[:,1]), distance=5, prominence=10)
    
        ## Calculation of the lines for visual inspection of the peaks' location:
        num_angles = len(peaks)
        dr=1
        profile_radius = 1.2*ves_coordinates[3]
        along_radius = np.arange(0, int(profile_radius), dr)
        
        line_x = np.zeros((len(along_radius), len(peaks)))
        line_y = np.zeros((len(along_radius), len(peaks)))
        
        peaks_radians = peaks*np.pi/180
    
        line_x = np.full((len(along_radius), num_angles), size_box) + np.rint(np.matmul(np.transpose(np.asmatrix(along_radius)),np.asmatrix(np.cos(peaks_radians))))
        line_y = np.full((len(along_radius), num_angles), size_box) + np.rint(np.matmul(np.transpose(np.asmatrix(along_radius)),np.asmatrix(np.sin(peaks_radians))))
            
        ## Calculation of a moving average to smooth the the angular profile:
        window_size = 30
        i = 0
        moving_averages = []
        
        while i < len(angular_profiles[:,1]) - window_size + 1:
            window_average = round(np.sum(angular_profiles[i:i+window_size,1]) / window_size, 2)
            moving_averages.append(window_average)
            i += 1
        
        ## Detection of peaks in the smoothed profile to test for improvements:
        peaks_smooth, _ = find_peaks(moving_averages, height=5, distance=5, prominence=1) 
        peaks_smooth = peaks_smooth + window_size/2
        
        ## Calculation of the lines for visual inspection of the peaks' location
        ## from the smoothed detection:
            
        num_angles = len(peaks_smooth)
        dr=1
        profile_radius = 1.2*ves_coordinates[3]
        along_radius = np.arange(0, int(profile_radius), dr)
        
        line_x_smooth = np.zeros((len(along_radius), len(peaks_smooth)))
        line_y_smooth = np.zeros((len(along_radius), len(peaks_smooth)))
        
        peaks_radians_smooth = peaks_smooth*np.pi/180
    
        line_x_smooth = np.full((len(along_radius), num_angles), size_box) + np.rint(np.matmul(np.transpose(np.asmatrix(along_radius)),np.asmatrix(np.cos(peaks_radians_smooth))))
        line_y_smooth = np.full((len(along_radius), num_angles), size_box) + np.rint(np.matmul(np.transpose(np.asmatrix(along_radius)),np.asmatrix(np.sin(peaks_radians_smooth))))
    
        corrected_average = (np.average(angular_profiles[:,1], weights=angular_profiles[:,1]<mean + (mean-np.min(moving_averages))))
    
    ## Plotting of the calculated values for visual inspection:
    # if rsd > 0.8:
    fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1], dpi=150)
    axes["A"].axhline(y = mean, c = 'gold', ls="--",label="mean")
    axes["A"].axhline(y = median, c = 'red', ls="--")
    axes["A"].axhline(y = corrected_average , c = 'cyan', ls="-.", label="median")
    if rsd > 0.8:
        axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id), color = "red")
        ## MOVING AVERAGE:
        axes["A"].plot(np.arange(window_size/2-1, len(moving_averages)+window_size/2-1), moving_averages, color='lawngreen', label = 'Moving average')
        ## PEAKS:------      --------        ---------      ---------     ---------
        # for j in range(len(peaks)):
        #     axes["A"].axvline(x = peaks[j], c = 'red', ls="--")
        # for j in range(len(peaks_smooth)):
        #     axes["A"].axvline(x = peaks_smooth[j], c = 'lawngreen', ls="--")
    else:
        axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id))
    # axes["A"].plot(theta*180/np.pi, angular_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,1], c = 'magenta', alpha=0.3 ,label = 'septin signal')
    
    axes["A"].legend(loc="upper left", fontsize=8)
    axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'I (a.u.)')
    axes["B"].set_title("Septin channel", fontsize = 10, pad =-1)
    axes["B"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced")
    axes["B"].set_axis_off()
    axes["C"].set_title("Detected peaks", fontsize = 10, pad =-1)
    axes["C"].imshow(image_box[:,:,1], cmap = "cmap_magenta_enhanced")
    if rsd > 0.8:
        ## ORIGINAL PEAKS:------      --------        ---------      ---------    
        for j in range(len(peaks)):
            axes["C"].plot(line_x, line_y, color = "red")
        ## SMOOTH PEAKS:------      --------        ---------      ---------    
        for j in range(len(peaks)):
            axes["C"].plot(line_x_smooth, line_y_smooth, color = "lawngreen")
    axes["C"].set_axis_off()
    axes["D"].set_axis_off()
    axes["D"].text(-0.2,0.3,"RSD            :  " + str(round(rsd,2)), size = 10)
    # axes["D"].text(-0.2,0.0,"Localization:  " + str(round(localization[0],3)), size = 10)
    axes["D"].text(-0.2,0.0,"Minimum:  " + str(round(np.min(angular_profiles[:,1]),3)), size = 10)
    axes["D"].text(-0.2,-0.3,"Median:  " + str(round(median,3)), size = 10)
    axes["D"].text(-0.2,-0.6,"New mean:  " + str(round(corrected_average,3)), size = 10)

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
        
    eexp_info : list
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
        
    fig, axes = plt.subplot_mosaic("AB;AC;AD;AE", width_ratios = [3, 1], dpi=150)
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

    fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1], dpi=150)
    axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,1], c = 'magenta', label = 'septin')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,2], c = 'gold', label = 'actin')
    axes["A"].legend(loc="upper left")
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
        
    fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1], dpi=150)
    axes["A"].set_title('Intensity radial profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(along_radius, radial_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(along_radius, radial_profiles[:,1], c = 'magenta', label = 'septin')
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
    axes["D"].text(-0.2,0.7,"Background M:  " + str(round(background[0,0],4)), size = 10)
    axes["D"].text(-0.2,0.5,"Background S:   " + str(round(background[0,1],4)), size = 10)
    axes["D"].text(-0.2,0.0,"Localization S :  " + str(round(localization[0],3)), size = 10)
    axes["D"].set_axis_off()
    
    fig.savefig(path_to_output+"\\" + exp_info[3] + "-Vesicle_" + str(vesicle_id) + "-Int_prof_Radial.png")
    
    ## Intensity angular profile:

    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [3, 1], dpi=150)
    axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id))
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,0], c = 'cyan', label = 'membrane')
    axes["A"].plot(theta*180/np.pi, angular_profiles[:,1], c = 'magenta', label = 'septin')
    axes["A"].legend(loc="upper left", fontsize=8)
    axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'I (a.u.)')
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
        
    fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1], dpi=150)
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
        
    fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [3, 1], dpi=150)
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
    
