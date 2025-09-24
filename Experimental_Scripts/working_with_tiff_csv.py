# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:13:15 2022

@author: Katerina
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap           # to create colormaps 
import pandas as pd                                                             # to open csv
from PIL import Image                                                           # to open image as tiff
from skimage import io                                                          # to open stack
import imageio.v2 as im                                                         # to open stack (alternative)
   

### OPEN TIFF FILES ---------------------------------------------------------------------------------------------------

## OPEN SINGLE IMAGE ----------------------------------------------------------
## To open a tiff image in python: (requires PIL)
    
im_membrane = Image.open('20220504_suzanne_ch01.tif')
im_septin = Image.open('20220504_suzanne_ch00.tif')

##To display the tiff images: 

im_membrane.show()
im_septin.show()        

## To open the image in TEXT format:

txt_membrane = np.loadtxt('Membrane_text_3.txt')
txt_septin   = np.loadtxt('Septin_text_3.txt')

## OPEN STACK OF IMAGES (requires skimage OR imageio.v2) ----------------------

# from skimage import io

im_stack_septin   = io.imread('20220915_sept_oct_ch00.tif')
im_stack_membrane = io.imread('20220915_sept_oct_ch01.tif')
print(im_stack_septin.shape)

## Alternative: To open a tiff image as a stack in python: 

# import imageio.v2 as im
im_stack_septin_2 = im.volread('20220915_sept_oct_ch00.tif')  # 
im_stack_membrane_2 = im.volread('20220915_sept_oct_ch01.tif')
print(im_stack_septin_2.shape)

### SHOW IMAGES IN PYTHON -------------------------------------------------------------------------------------------------

## Defining colormaps that match the LUTs of ImageJ:
colors=["black","cyan"]
nodes = [0.0, 1.0]
my_cmap_cyan = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

colors=["black","magenta"]
nodes = [0.0, 1.0]
my_cmap_magenta = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

## Show single image:
plt.figure()
plt.axis('off')
plt.style.use('dark_background')
plt.imshow(im_stack_membrane[12,:,:], cmap = my_cmap_cyan)
plt.show()
plt.axis('off')
plt.style.use('dark_background')
plt.imshow(im_stack_septin[12,:,:], cmap = my_cmap_magenta)
plt.show()

## Show image with overlay:
    
import napari

img_2chan = np.zeros((len(txt_membrane[:,0]), len(txt_membrane[0,:]), 2))
img_2chan[:,:,0] = txt_septin
img_2chan[:,:,1] = txt_membrane
print(img_2chan.shape)

viewer = napari.Viewer()

viewer.add_image(img_2chan[..., 0], colormap="magenta", name="Channel 1", blending="additive")
viewer.add_image(img_2chan[..., 1], colormap="cyan", name="Channel 2", blending="additive")


## Show stack:

for t in range(0,len(im_stack_membrane[:,0,0])):
    plt.figure(dpi = 220)
    plt.imshow(im_stack_membrane[t,:,:], cmap = my_cmap_cyan)
    plt.title("Membrane channel, Z-stack: " + str(t+1) + '/' + str(len(im_stack_membrane[:,0,0])))
    plt.axis("off")
    #plt.savefig("./img/Folder_to_save" + str(t)+ '/' + str(len(im_stack_membrane[:,0,0])) + ".png", dpi = 250)
    plt.show()

## To open csv file and turn it into an array: (requires pd)
## NOTE: disGUVery saved the first column as '# id_vesicle' and '#' seems to give an issue

detection_ch    = pd.read_csv(r'Composite_detected_vesicles_1.csv')
detection_array = pd.DataFrame(detection_ch).to_numpy()    





    

    