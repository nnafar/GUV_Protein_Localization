# -*- coding: utf-8 -*-
"""
intensity_profiles.py
-------------------------------------------------------------------------------
This code prints the radial and angular intensity profiles of vesicles detected 
by DisGUVery. It accounts for background noise and quantifies localization on 
the membrane.

At the current version the code considers only the membrane and the septin channels. 
Further modifications are needed to also accommodate for actin. 

For convenience certain parts of the code should be turned into functions at a 
later stage of development. 

Input
-------------------------------------------------------------------------------
- The path to the text images of the channels should always be given by the user.
  NOTE: The channels should be stored and loaded as text files
- The path to the detected_vesicles output csv file of DisGUVery should always 
  be given by the user. 
  NOTE: The '#' at the beginning of the first column's title in the csv file must
        be manually deleted by the user before running the code. In case of further
        problems consider removing the ' ' in front of the letters of the first 
        column's title as well. 

Short description
-------------------------------------------------------------------------------
1. The text images and the csv are read. The csv columns are turned into separate
   arrays for convenience. 
2. All parameter values are defined by the user.
3. Linear profiles are calculated for different angles. The profiles start from 
   the detected centre of the vesicles and extend a bit further than the detected 
   radius to account for inaccuracies in the detection and deformations of the 
   membrane. The number of linear profiles per vesicle and the excess length are
   defined by the user.
   NOTE: If the excess length ends up being outside the image the vesicle is 
         not considered further (Too close to the membrane)
4. The background noise is calculated and subtracted from the linear profiles.
   Negative values occurring after the subtraction are replaced by 0.
   - For the membrane channel: The user defines the size of the area that includes
                               the vesicle to be considered. Local thresholding 
                               (skimage) is applied and a mask is created. After 
                               filling in the inside of the vesicle the mask is 
                               applied for avearaging the intensity excluding the
                               areas of high signal for the estimation of the 
                               background.
   - For the septin channel:   A mask is created the same way as in the membrane
                               channel in case of non-encapsulated protein clusters.
                               This mask is added to the membrane mask and used 
                               for avearaging the intensity excluding the areas
                               of high signal for the estimation of the background.               
5. By averaging all the line profiles for all the angles a collective radial 
   profile is obtained for each vesicle. 
6. Membrane detection takes place, using scipy's find_pekas and considering
   a membrane width equal to the width at half maximum of the peak. 
7. Localization is defined as the average value of the protein intensity within
   the detected membrane area over the average value of the protein intensity 
   from the centre of the vesicle up to 1/4 of the vesicle radius. 
8. The angular intensity profile is calculated by averaging along the radial direction
   within the detected membrane disc for every angle.
9. Plotting intensity profiles:
   The radial intensity profile is printed along with a zoomed in image of the 
   vesicle in all channels for convenience of data interpretation. The background
   noise found for all channels and the quantified localization are also printed.
   (At the current version the detected membrane borders and the border of the 
    considered central area of the vesicle are also marked on the radial profile)
   The angular intensity profile is printed along with a zoomed in image of the 
   vesicle in all channels for convenience of data interpretation.

Parameters
-------------------------------------------------------------------------------
num_angles : int
       The number of the linear profiles to be taken into account for the 
       calculation of the collected intensity profile of the given vesicle.
length_excess : float
       The factor multiplying the radius of the given vesicle in the analysis.
       Meant to extend the linear profiles further than the detected radius to
       account for inaccuracies in the detection and membrane deformations and 
       asymmetries.
dr : int
       The step (in px) between the points sampled along the radial direction
       for the linear profiles calculation.  
size_factor_mask  : float 
       The factor that will multiply the vesicle radius to determine the 
       length of the square side to be considered during background noise
       calculation.
size_factor_view  : float 
       The factor that will multiply the vesicle radius to determine the
       length of the square side to be viewed for visual inspection along 
       the final plots.
threshold_method : str
       The threshold method chosen by the user

Output
-------------------------------------------------------------------------------
The current version plots the radial and angular intensity profiles and 
quantifies the localization of the protein on the membrane. Optionally, the 
intensity profiles can also be normalized. 

-------------------------------------------------------------------------------     
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                                                            # to access csv file 
from matplotlib.colors import LinearSegmentedColormap                          # to create colormaps matching our conventions 

from skimage.filters import threshold_local                                    # to find threshold and create mask 
from skimage.filters import try_all_threshold
from skimage.filters import threshold_isodata, threshold_li, threshold_mean, threshold_minimum, threshold_otsu, threshold_triangle, threshold_yen
                                  
from scipy import ndimage as nd                                                # to fill in the mask contours
from scipy.signal import chirp, find_peaks, peak_widths

### 1. READING FILES ----------------------------------------------------------

## Reading channels from text image files:

txt_membrane = np.loadtxt('1-20221026_sept_oct_different_comp_0%DOPS_0%PIP2_Region1_Membrane_channel.txt')
txt_septin   = np.loadtxt('1-20221026_sept_oct_different_comp_0%DOPS_0%PIP2_Region1_Septin_channel.txt')

## Reading csv file of detected vesicles:                                      # IMPORTANT NOTE: manually delete '#' from first column's title: '# id_vesicle' in DisGUVery output file

detection_ch = pd.read_csv(r'1-20221026_sept_oct_different_comp_0%DOPS_0%PIP2_Region1_detected_vesicles.csv')

identity     = pd.DataFrame(detection_ch, columns= [' id_vesicle']).to_numpy()
xc           = pd.DataFrame(detection_ch, columns= [' xc (pix)']).to_numpy()
yc           = pd.DataFrame(detection_ch, columns= [' yc (pix)']).to_numpy()
radius       = pd.DataFrame(detection_ch, columns= [' radius (pix)']).to_numpy()

num_vesicles = len(radius)

## Defining colormaps that match the LUTs of ImageJ used conventionally:
    
colors          = ["black", "cyan"]
nodes           = [0.0, 1.0]
my_cmap_cyan    = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

colors          = ["black","magenta"]
nodes           = [0.0, 1.0]
my_cmap_magenta = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))


## Optional plotting of the membrane channel with annotated centres of vesicles:
#  NOTE: Recommended when troubleshooting

plot_detected_centres = 'True'

if plot_detected_centres == 'True':
    plt.figure()
    plt.title('Detected vesicles from DisGUVery: ' + str(num_vesicles))   
    plt.axis('off')
    plt.style.use('default')    # alternative viewing: 'dark_background'
    plt.imshow(txt_membrane, cmap=my_cmap_cyan)
    plt.scatter(xc, yc, color='red', s=1, marker="o")
    for i in range(num_vesicles):
        plt.annotate(i+1, (xc[i]+0, yc[i]-0), c = 'red') 
    plt.show()

## Will be adjusted: Review and choice of threshold method:   

image_dim_x    = len(txt_membrane[0,:])
image_dim_y    = len(txt_membrane[:,0])

vesicle_box = 2*radius[0]

move_right = 0
move_left  = 0
move_up    = 0
move_down  = 0

if xc[0]- vesicle_box < 0: 
    move_right = vesicle_box - xc[0]           
if xc[0]+vesicle_box > image_dim_x: 
    move_left  = vesicle_box - (image_dim_x-xc[0])    
if yc[0]-vesicle_box < 0: 
    move_down  = vesicle_box - yc[0]                        
if yc[0]+vesicle_box > image_dim_y: 
    move_up    = vesicle_box - (image_dim_y-yc[0])   

vesicle_box_memb = txt_membrane[int(yc[0]-vesicle_box+move_down):int(yc[0]+vesicle_box-move_up), int(xc[0]-vesicle_box+move_right):int(xc[0]+vesicle_box-move_left)]

fig, ax = try_all_threshold(vesicle_box_memb, figsize=(5, 10), verbose=False)
plt.show()                  

threshold_method = input('Choose threshold method to follow:')

### 2. DEFINING PARAMETER VALUES ----------------------------------------------

num_angles       = 360                                                         # number of linear profiles to be taken into account
length_excess    = 1.6                                                         # the factor multiplying the radi of the vesicles for the linear profiles to account for deformations
dr               = 1                                                           # the step (in px) between the points sampled along the radial direction for the linear profiles calculation  
size_factor_mask = 3                                                           # the factor multiplying the radi of each vesicle for the length of the square side to be considered during background noise calculation
size_factor_view = length_excess                                               # the factor multiplying the radi of each vesicle for the length of the square side used for visual inspection along the final graphs
# might have to add center_relative_size = 1/4

def intensity_profile(vesicle_id, radius, xc, yc, num_angles, length_excess, dr, size_factor_mask, size_factor_view, txt_membrane, txt_septin, threshold_method):
    """
    Function calculating the radial and angular intensity profile of a single
    given vesicle.
    NOTES: - Should be broken in smaller functions for convenience.
            - At the current version it prints the plots calculated. This should
              be changed to giving a certain output.

    Parameters
    ----------
    vesicle_id : int
        The identity of the vesicle of which the intensity profile should be 
        plotted.
    radius : float
        The radius of the given vesicle.
    xc : float
        The index of the center of the given vesicle in the original text image.
        NOTE: x coordinate corresponds to columns of the text image array
    yc : float
        The index of the center of the given vesicle in the original text image.
        NOTE: y coordinate corresponds to rows of the text image array 
              e.g. yc = 0 : top row 
    num_angles : int
        The number of the linear profiles per vesicle to be taken into account for
        the calculation of the collected intensity profile of the given vesicle.
    length_excess : float
        The factor multiplying the radius of the given vesicle in the analysis.
        Meant to extend the linear profiles further than the detected radius to
        account for inaccuracies in the detection and membrane deformations and 
        asymmetries.
    dr : int
        The step (in px) between the points sampled along the radial direction
        for the linear profiles calculation.
   size_factor_mask  : float 
        The factor that will multiply the vesicle radius to determine the 
        length of the square side to be considered during background noise
        calculation.
    size_factor_view  : float 
        The factor that will multiply the vesicle radius to determine the
        length of the square side to be viewed for visual inspection along 
        the final plots.
    txt_membrane : np.ndarray
        The text image of the membrane channel.
    txt_septin : np.ndarray
        The text image of the septin channel.
    threshold_method : str
        The threshold method chosen by the user

    Returns
    -------
    In the current version it does not return anything. It plots the radial and 
    angular intensity profiles after background subtraction and quantifies the 
    localization on the membrane. 
    NOTE: Should be changed to giving an output to be used for subsequent plots.
          
    """

    ### 3. CALCULATING LINEAR PROFILES ----------------------------------------
    
    ## Determining image dimensions:
    image_dim_x    = len(txt_membrane[0,:])
    image_dim_y    = len(txt_membrane[:,0])
    
    ## Defining angles and sampling points for the linear profiles:
        
    theta        = np.linspace(0, 2*np.pi, num_angles, endpoint=False)         # chosen as such so that the lines are spread uniformly covering the whole circle and without double counting of 0 and 2*pi 
    along_radius = np.arange(0, int(length_excess*radius), dr)                 # defining sampling points along the linear profile corresponding to 0 degrees with starting point the detected center of the given vesicle
    
    ## Defining the image indices corresponding to the pixels along the linear profiles:
    #  NOTE: rounding to nearest integer is used in contributing pixel decision making
    
    line_x       = np.full((len(along_radius), num_angles), int(xc)) + np.rint(np.matmul(np.transpose(np.asmatrix(along_radius)),np.asmatrix(np.cos(theta))))
    line_y       = np.full((len(along_radius), num_angles), int(yc)) + np.rint(np.matmul(np.transpose(np.asmatrix(along_radius)),np.asmatrix(np.sin(theta))))
        
    ## Assigning the intensity values of the channels corresponding to the previously defined lines in arrays:
    #  NOTE: Vesicles that are partially in the image or vesicles for which the excess radius leads out of the image are disregarded.  
    intensity_membrane = np.zeros((len(along_radius), num_angles))
    intensity_septin   = np.zeros((len(along_radius), num_angles)) 
    
    for i in range(len(along_radius)):
        for j in range((num_angles)):            
            
            ## Checking if the excess radius leads outside of the image borders:
            #  NOTE: In that case (or if vesicle is cut at the border) the vesicle is disregarded. 
            
            if xc + length_excess*radius < image_dim_x and xc - length_excess*radius >= 0 and yc + length_excess*radius < image_dim_y and yc - length_excess*radius >= 0:
                intensity_membrane[i,j] = txt_membrane[int(line_y[i,j]), int(line_x[i,j])]
                intensity_septin[i,j]   = txt_septin[int(line_y[i,j]), int(line_x[i,j])]
            else:
                return
           
    ### 4. CALCULATING BACKGROUND NOISE ---------------------------------------
    
    ## Method used: Threshold and Mask
    # Description:  Creates mask over the vesicle and over surrounding lipid aggregates 
    #               Average over a zoomed-in image centered at the vesicle avoiding
    #               the high inthensity points. 
    
    ## Defining size of the area around the vesicle considered for the background calculation:
    
    vesicle_box =  size_factor_mask * radius

    ## Creating misplacment parameters so that the vesicle box remains within the image borders: 
    #  NOTE: Necessary only for vesicles close to the borders
    
    move_right = 0
    move_left  = 0
    move_up    = 0
    move_down  = 0
    
    if xc-vesicle_box < 0: 
        move_right = vesicle_box - xc           
    if xc+vesicle_box > image_dim_x: 
        move_left  = vesicle_box - (image_dim_x-xc)    
    if yc-vesicle_box < 0: 
        move_down  = vesicle_box - yc                        
    if yc+vesicle_box > image_dim_y: 
        move_up    = vesicle_box - (image_dim_y-yc)                     
     
    ###-----------------Threshold and Mask ------------------------------------
    
    ## Defining of the area we focus on for background calculation:
        
    vesicle_box_memb = txt_membrane[int(yc-vesicle_box+move_down):int(yc+vesicle_box-move_up), int(xc-vesicle_box+move_right):int(xc+vesicle_box-move_left)]
    vesicle_box_sept = txt_septin[int(yc-vesicle_box+move_down):int(yc+vesicle_box-move_up), int(xc-vesicle_box+move_right):int(xc+vesicle_box-move_left)]
    
    ## Thresholding and creating mask
     
    ## Choice of threshold method:
        
    if threshold_method == 'Isodata':
        thresh_memb = threshold_isodata(vesicle_box_memb)
        thresh_sept = threshold_isodata(vesicle_box_sept)
    elif threshold_method == 'Li':
        thresh_memb = threshold_li(vesicle_box_memb)
        thresh_sept = threshold_li(vesicle_box_sept)
    elif threshold_method == 'Mean':
        thresh_memb = threshold_mean(vesicle_box_memb)
        thresh_sept = threshold_mean(vesicle_box_sept)
    elif threshold_method == 'Minimum':
        thresh_memb = threshold_minimum(vesicle_box_memb)
        thresh_sept = threshold_minimum(vesicle_box_sept)
    elif threshold_method == 'Otsu':
        thresh_memb = threshold_otsu(vesicle_box_memb)
        thresh_sept = threshold_otsu(vesicle_box_sept)
    elif threshold_method == 'Triangle':
        thresh_memb = threshold_triangle(vesicle_box_memb)
        thresh_sept = threshold_triangle(vesicle_box_sept)
    elif threshold_method == 'Yen':
        thresh_memb = threshold_yen(vesicle_box_memb)
        thresh_sept = threshold_yen(vesicle_box_sept)
    else:
        print('Invalid method!')
        return
    
    # Creating membrane mask
    mask_memb = vesicle_box_memb > thresh_memb
    
    # Creating assistive mask for the septin channel in case of non-encapsulated clusters present in the picture
    mask_sept_assist = vesicle_box_sept > thresh_sept

    ## Filling in the vesicles:
    mask_memb_fill = nd.binary_fill_holes(mask_memb).astype(int)
    
    ## Filling in vesicles cut at the borders
    border_cuts_left, = np.where(mask_memb[:,0] == 1)
    
    if len(border_cuts_left[:]) > 2:
        assistant_left = np.zeros((len(mask_memb[:,0]),1))
        assistant_left[border_cuts_left[0]:border_cuts_left[-1]] = 1
        mask_left = np.hstack((assistant_left, mask_memb_fill))
        mask_fill_left = nd.binary_fill_holes(mask_left).astype(int)
        mask_memb_fill = mask_fill_left[:,1:]
       
    border_cuts_right, = np.where(mask_memb[:,-1] == 1)
    
    if len(border_cuts_right[:]) > 2:
        assistant_right = np.zeros((len(mask_memb[:,-1]),1))
        assistant_right[border_cuts_right[0]:border_cuts_right[-1]] = 1
        mask_right = np.hstack((mask_memb_fill, assistant_right))
        mask_fill_right = nd.binary_fill_holes(mask_right).astype(int)
        mask_memb_fill = mask_fill_right[:,:-1]
    
    border_cuts_up, = np.where(mask_memb[0,:] == 1)
    
    if len(border_cuts_up[:]) > 2:
        assistant_up = np.zeros((1,len(mask_memb[0,:])))
        assistant_up[border_cuts_up[0]:border_cuts_up[-1]] = 1
        mask_up = np.vstack((assistant_up, mask_memb_fill))
        mask_fill_up = nd.binary_fill_holes(mask_up).astype(int)
        mask_memb_fill = mask_fill_up[1:,:]
        
    border_cuts_down, = np.where(mask_memb[-1,:] == 1)
    
    if len(border_cuts_down[:]) > 2:
        assistant_down = np.zeros((1,len(mask_memb[0,:])))
        assistant_down[border_cuts_down[0]:border_cuts_down[-1]] = 1
        mask_down = np.vstack((mask_memb_fill, assistant_down))
        mask_fill_down = nd.binary_fill_holes(mask_down).astype(int)
        mask_memb_fill = mask_fill_down[:-1,:]
    
    # Septin mask is taken as the membrane mask adding the assistive septin mask in case of non-encapsulated septin aggregates
    mask_sept = np.where(mask_memb_fill + mask_sept_assist != 2, mask_memb_fill + mask_sept_assist, 1)
    
    ## Averaging using the mask for the background calculation:
        
    background_membrane = np.mean(np.ma.array(vesicle_box_memb, mask=mask_memb))
    background_septin   = np.mean(np.ma.array(vesicle_box_sept, mask=mask_sept))
    
    ## Optional plotting of images to assist quality of masks:
    #  NOTE: Recommended during troubleshooting
    
    plot_masks = 'True'
    
    if plot_masks == 'True':
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,4))
        fig.suptitle('Local region of Vesicle ' + str(vesicle_id+1))
        ax1.imshow(vesicle_box_memb , cmap = my_cmap_cyan)
        ax1.set_title('Membrane')
        ax1.set_axis_off()
        ax2.imshow(vesicle_box_sept , cmap = my_cmap_magenta)
        ax2.set_title('Septin')
        ax2.set_axis_off()
        plt.show()
        
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,4))
        fig.suptitle('Mask of Vesicle ' + str(vesicle_id+1))
        ax1.imshow(mask_memb , cmap = my_cmap_cyan)
        ax1.set_title('Membrane')
        ax1.set_axis_off()
        ax2.imshow(mask_sept_assist , cmap = my_cmap_magenta)
        ax2.set_title('Septin')
        ax2.set_axis_off()
        plt.show()
        
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,4))
        fig.suptitle('Filled mask of Vesicle ' + str(vesicle_id+1))
        ax1.imshow(mask_memb_fill , cmap = my_cmap_cyan)
        ax1.set_title('Membrane')
        ax1.set_axis_off()
        ax2.imshow(mask_sept , cmap = my_cmap_magenta)
        ax2.set_title('Septin')
        ax2.set_axis_off()
        plt.show()    


    ## Subtracting background from linear profiles:
        
    intensity_membrane_corrected = np.clip(intensity_membrane - background_membrane, 0, None) # setting negative values occuring after subtraction to 0
    intensity_septin_corrected   = np.clip(intensity_septin - background_septin, 0, None) 

    ### 5. COLLECTIVE RADIAL PROFILE ------------------------------------------
    
    ## Averaging the linear intensity profiles over all considered angles:
        
    mean_int_membr = np.average(intensity_membrane_corrected, axis = 1)
    mean_int_sept  = np.average(intensity_septin_corrected, axis = 1) 

    ### 6. MEMBRANE DETECTION -------------------------------------------------
    
    ## Finding peak:
    
    peaks, _ = find_peaks(mean_int_membr, height=10, distance=5, prominence=1)
    width_half_max = peak_widths(mean_int_membr, peaks, rel_height=0.5)
        
    index_border_in = int(np.rint(width_half_max[2][-1]))
    index_border_out = int(np.rint(width_half_max[3][-1]))
    
    ### 7. QUANTIFYING LOCALIZATION -------------------------------------------
    
    septin_bound   = np.average(mean_int_sept[index_border_in:index_border_out])
    septin_centre  = np.average(mean_int_sept[0:int(1/4*radius)])  
    # might change to: septin_centre  = np.average(mean_int_sept[0:int(centre_relative_size*radius)])
    
    localization   = septin_bound/septin_centre
    
    ### 8. ANGULAR PROFILE ----------------------------------------------------
        
    membrane_angular = np.average(intensity_membrane_corrected[index_border_in:index_border_out,:], axis=0)
    septin_angular   = np.average(intensity_septin_corrected[index_border_in:index_border_out,:], axis=0)
        
    ### 9. PLOTTING INTENSITY PROFILES ---------------------------------------- 
        
    ## Creating zoomed-in picture for conveninece:   
    # Defining size of the area for visual inspection along the results:
        
    zoom_in_box =  size_factor_view * radius
    
    ## Creating misplacment parameters so that the vesicle box remains within the image borders: 
    #  NOTE: Necessary only for vesicles close to the borders
        
    move_right = 0
    move_left  = 0
    move_up    = 0
    move_down  = 0
        
    if xc-zoom_in_box < 0: 
        move_right = zoom_in_box - xc           
    if xc+zoom_in_box > image_dim_x: 
        move_left  = zoom_in_box - (image_dim_x-xc)    
    if yc-zoom_in_box < 0: 
        move_down  = zoom_in_box - yc                        
    if yc+zoom_in_box > image_dim_y: 
        move_up    = zoom_in_box - (image_dim_y-yc)  
        
    ## Defining the area for visual inspection along the results:
            
    zoom_in_vesicle_membrane = txt_membrane[int(yc-zoom_in_box+move_down):int(yc+zoom_in_box-move_up), int(xc-zoom_in_box+move_right):int(xc+zoom_in_box-move_left)]
    zoom_in_vesicle_septin = txt_septin[int(yc-zoom_in_box+move_down):int(yc+zoom_in_box-move_up), int(xc-zoom_in_box+move_right):int(xc+zoom_in_box-move_left)]
    
    ## Plotting of results: 
        
    plot_final_results = 'True' 
    
    if plot_final_results == 'True': 
        
        ## INTENSITY RADIAL PROFILE:
            
        fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1])
        axes["A"].set_title('Intensity radial profile of Vesicle ' + str(vesicle_id+1))
        axes["A"].plot(along_radius, mean_int_membr, c = 'cyan', label = 'membrane')
        axes["A"].plot(along_radius, mean_int_sept, c = 'magenta', label = 'septin')
        axes["A"].axvline(x = along_radius[index_border_in], c = 'red', label = 'border in', ls="--")
        axes["A"].axvline(x = along_radius[index_border_out], c = 'orange', label = 'border out', ls="--")
        axes["A"].axvline(x = radius, c = 'black', ls=":")
        axes["A"].axvline(x = 1/4*radius[0], c ='green', label = 'central area border') ## if it's changed to a parameter, e.g. center_relative_size, change this too
        axes["A"].set(xlabel = 'radius (px)', ylabel = 'I (a.u.)')
        axes["A"].legend(loc="upper left")
        axes["B"].set_title("Membrane channel")
        axes["B"].imshow(zoom_in_vesicle_membrane, cmap = my_cmap_cyan)
        axes["B"].set_axis_off()
        axes["C"].set_title("Septin channel")
        axes["C"].imshow(zoom_in_vesicle_septin, cmap = my_cmap_magenta)
        axes["C"].set_axis_off()
        axes["D"].text(-0.2,0.7,"Background M:  " + str(round(background_membrane,4)), size = 10)
        axes["D"].text(-0.2,0.5,"Background S:   " + str(round(background_septin,4)), size = 10)
        axes["D"].text(-0.2,0.3,"Localization    :  " + str(round(localization,3)), size = 10)
        axes["D"].set_axis_off()
        plt.show()
   
        ## INTENSITY ANGULAR PROFILE:

        fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [3, 1])
        axes["A"].set_title('Intensity angular profile of Vesicle ' + str(vesicle_id+1))
        axes["A"].plot(theta*180/np.pi, membrane_angular, c = 'cyan', label = 'membrane')
        axes["A"].plot(theta*180/np.pi, septin_angular, c = 'magenta', label = 'septin')
        axes["A"].legend(loc="upper left")
        axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'I (a.u.)')
        axes["B"].set_title("Membrane channel")
        axes["B"].imshow(zoom_in_vesicle_membrane, cmap = my_cmap_cyan)
        axes["B"].set_axis_off()
        axes["C"].set_title("Septin channel")
        axes["C"].imshow(zoom_in_vesicle_septin, cmap = my_cmap_magenta)
        axes["C"].set_axis_off()
    
        plt.show()
        
    ## Optional plotting of normalized results: 
        
    plot_final_results_normalize = 'False' 
    
    if plot_final_results_normalize == 'True': 
        
        ## NORMALIZED INTENSITY RADIAL PROFILE:
            
        fig, axes = plt.subplot_mosaic("AB;AC;AD", width_ratios = [3, 1])
        axes["A"].set_title('Normalized intensity radial profile of Vesicle ' + str(vesicle_id+1))
        axes["A"].plot(along_radius, mean_int_membr/max(mean_int_membr) , c = 'cyan', label = 'membrane')
        axes["A"].plot(along_radius, mean_int_sept/max(mean_int_membr), c = 'magenta', label = 'septin')
        axes["A"].axvline(x = along_radius[index_border_in], c = 'red', label = 'border in')
        axes["A"].axvline(x = along_radius[index_border_out], c = 'orange', label = 'border out')
        axes["A"].axvline(x = 1/4*radius[0], c ='green', label = 'central area border') ## if it's changed to a parameter, e.g. center_relative_size, change this too
        axes["A"].set(xlabel = 'radius (px)', ylabel = 'I / $I_{max}$')
        axes["A"].legend(loc="upper left")
        axes["B"].set_title("Membrane channel")
        axes["B"].imshow(zoom_in_vesicle_membrane, cmap = my_cmap_cyan)
        axes["B"].set_axis_off()
        axes["C"].set_title("Septin channel")
        axes["C"].imshow(zoom_in_vesicle_septin, cmap = my_cmap_magenta)
        axes["C"].set_axis_off()
        axes["D"].text(-0.2,0.7,"Background M:  " + str(round(background_membrane,4)), size = 10)
        axes["D"].text(-0.2,0.5,"Background S:   " + str(round(background_septin,4)), size = 10)
        axes["D"].text(-0.2,0.3,"Localization    :  " + str(round(localization,3)), size = 10)
        axes["D"].set_axis_off()
        plt.show()
   
        ## NORMALIZED INTENSITY ANGULAR PROFILE:

        fig, axes = plt.subplot_mosaic("AB;AC", width_ratios = [3, 1])
        axes["A"].set_title('Normalized intensity angular profile of Vesicle ' + str(vesicle_id+1))
        axes["A"].plot(theta*180/np.pi, membrane_angular/max(membrane_angular), c = 'cyan', label = 'membrane')
        axes["A"].plot(theta*180/np.pi, septin_angular/max(membrane_angular), c = 'magenta', label = 'septin')
        axes["A"].legend(loc="upper left")
        axes["A"].set(xlabel = '\u03B8 (deg)', ylabel = 'I / $I_{max} (a.u.)')
        axes["B"].set_title("Membrane channel")
        axes["B"].imshow(zoom_in_vesicle_membrane, cmap = my_cmap_cyan)
        axes["B"].set_axis_off()
        axes["C"].set_title("Septin channel")
        axes["C"].imshow(zoom_in_vesicle_septin, cmap = my_cmap_magenta)
        axes["C"].set_axis_off()
    
        plt.show()

## Calling the function for all detected vesicles of given image:        
for vesicle_id in range(num_vesicles):
    intensity_profile(vesicle_id, radius[vesicle_id], xc[vesicle_id], yc[vesicle_id], num_angles, length_excess, dr, size_factor_mask, size_factor_view, txt_membrane, txt_septin, threshold_method) 



