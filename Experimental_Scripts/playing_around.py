import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import cv2

# .ListedColormap as ListedColormap

# reading
stack_3chan = iio.volread('data/20220915_act_sept.tif')
plt.figure(figsize=(15, 15))
plt.imshow(stack_3chan[21][0])  # membrane
plt.show()
plt.imshow(stack_3chan[21][1])  # septin
plt.show()
plt.imshow(stack_3chan[21][2])  # actin
plt.show()

stack_frame21 = iio.volread('data/frame21.tif')
plt.figure(figsize=(15, 15))
plt.imshow(stack_frame21[0])  # septin
plt.show()
plt.imshow(stack_frame21[1])  # actin
plt.show()
plt.imshow(stack_frame21[2])  # membrane
plt.show()

# comp = iio.imread('C:\\Users\\Elise\\Documents\\MEP\\Data\\20220922_wt_test\\20220915_act_sept_im1')

# split up into channels
septin = stack_frame21[0]
actin = stack_frame21[1]
membrane = stack_frame21[2]

# numpy arrays?
septin_np = np.asarray(septin)
actin_np = np.asarray(actin)
membrane_np = np.asarray(membrane)

# try show with different colors
plt.imshow(septin_np, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('magentamap', ["black", "magenta"]))
plt.show()

# cmapmagenta = matplotlib.colors.LinearSegmentedColormap.from_list('magentamap', ["magenta", "black"])

plt.imshow(actin_np, cmap = matplotlib.colors.LinearSegmentedColormap.from_list('yellowmap', ["black", "yellow"]))
plt.show()

plt.imshow(membrane_np, cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cyanmap', ["black", "cyan"]))
plt.show()

im0 = plt.imshow(septin_np, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('magentamap', ["black", "magenta"]))
im1 = plt.imshow(actin_np, cmap = matplotlib.colors.LinearSegmentedColormap.from_list('yellowmap', ["black", "yellow"]), alpha=.4)
im2 = plt.imshow(membrane_np, cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cyanmap', ["black", "cyan"]), alpha=.4)
plt.show()
# probably not quite how you should be doing it, but seems to work kinda

# import stuff from disguvery
detected_vesicles = pd.read_csv("C:\\Users\\Elise\\Documents\\MEP\\Data\\20220915_guvs_actin_oct9i1\\20220915_act_sept_im1\\frame21_detected_vesicles.csv")
results_angular_profiles = pd.read_csv("C:\\Users\\Elise\\Documents\\MEP\\Data\\20220915_guvs_actin_oct9i1\\20220915_act_sept_im1\\frame21_results_angular_profiles.csv")
results_radial_profiles = pd.read_csv("C:\\Users\\Elise\\Documents\\MEP\\Data\\20220915_guvs_actin_oct9i1\\20220915_act_sept_im1\\frame21_results_radial_profiles.csv")

# make it numpy
detected_vesicles_np = pd.DataFrame.to_numpy(detected_vesicles)
results_angular_profiles_np = pd.DataFrame.to_numpy(results_angular_profiles)
results_radial_profiles_np = pd.DataFrame.to_numpy(results_radial_profiles)

# visualize detected vesicles (Hough) - different type if different detection?
figure_circles, axes_circles = plt.subplots()
plt.scatter(detected_vesicles_np[:,1],detected_vesicles_np[:,2],c='red')
for i in range(len(detected_vesicles_np)):
    plt.annotate(i+1, (detected_vesicles_np[i,1], detected_vesicles_np[i,2]), c='red')
    circle = plt.Circle((detected_vesicles_np[i,1], detected_vesicles_np[i,2]), detected_vesicles_np[i,3], fill=False, color='red')
    axes_circles.add_patch(circle)
plt.imshow(membrane_np, origin='upper', cmap=matplotlib.colors.LinearSegmentedColormap.from_list('magentamap', ["black", "cyan"]))
plt.show()
plt.figure(figsize=(15,15))
plt.imshow(septin_np, origin='upper', cmap=matplotlib.colors.LinearSegmentedColormap.from_list('magentamap', ["black", "magenta"]))
plt.show()

# plot results angular profiles first vesicle
stepsize_theta = 2
figure_angularpr, axes_angularpr = plt.subplots()
plt.plot(results_angular_profiles_np[0:int(360/stepsize_theta-1),2], results_angular_profiles_np[0:int(360/stepsize_theta-1),3], color='magenta', label='septin')
plt.plot(results_angular_profiles_np[int(360/stepsize_theta): int(360*2/stepsize_theta-1),2], results_angular_profiles_np[int(360/stepsize_theta): int(360*2/stepsize_theta-1),3], color='yellow', label='actin')
plt.plot(results_angular_profiles_np[int(2*360/stepsize_theta): int(360*3/stepsize_theta-1),2], results_angular_profiles_np[int(360*2/stepsize_theta): int(360*3/stepsize_theta-1),3], color='cyan', label='lipids')
plt.legend()
axes_angularpr.set_xlabel('angle')
axes_angularpr.set_ylabel('mean intensity')
plt.show()
# for loop, may want to use forloop over max vesicle id; include vesicle id in header

# plot results radial profiles first vesicle
range = int(len(results_radial_profiles_np)/(max(results_radial_profiles_np[:,0]*max(results_radial_profiles_np[:,1]))))
figure_radialpr, axes_radialpr = plt.subplots()
plt.plot(results_radial_profiles_np[0:range-1,2], results_radial_profiles_np[0:range-1,3], color='magenta', label='septin')
plt.plot(results_radial_profiles_np[range: 2*range-1,2], results_radial_profiles_np[range: 2*range-1,3], color='yellow', label='actin')
plt.plot(results_radial_profiles_np[2*range: 3*range-1,2], results_radial_profiles_np[2*range: 3*range-1,3], color='cyan', label='lipids')
plt.legend()
axes_radialpr.set_xlabel('radius (pixels)')
axes_radialpr.set_ylabel('mean intensity')
plt.show()

# quantify relative amount signal membrane vs middle - maybe consider splitting up data so that the values are contained at the same locations?
# localize membrane: find max, use full width half max (robust?)
max = np.amax(results_radial_profiles_np[2*range:-1,3]) #cheeting a little bit due to imperfect data
locmax = 2*range + np.argmax(results_radial_profiles_np[2*range:-1,3])
locleftedge = 2*range + np.argmin(np.abs(results_radial_profiles_np[2*range:locmax,3]-max/2))
locrightedge = locmax + np.argmin(np.abs(results_radial_profiles_np[locmax:,3]-max/2))
# build in some control? eg check distance edges not too big, not too far from center, avg intensity inbetween high

# determine avg values septin and actin at membrane region
sept_mem_avg = np.mean(results_radial_profiles_np[locleftedge-2*range:locrightedge-2*range,3])
act_mem_avg = np.mean(results_radial_profiles_np[locleftedge-range:locrightedge-range,3])

# determine avg values septin and actin outside membrane region; take avg over region 0-max min 2*dist til halve max
sept_enc_avg = np.mean(results_radial_profiles_np[:locmax-2*(locmax-locleftedge)-2*range,3])
act_enc_avg = np.mean(results_radial_profiles_np[range:-range+locmax-2*(locmax-locleftedge),3])

# determine ratio
sept_ratio = sept_mem_avg/sept_enc_avg
act_ratio = act_mem_avg/act_enc_avg

# scatterplot encapsulation septin vs actin?



#%%
# Image multiple vesicles

#%%
# background correction by subtraction 'empty' region

stack_background = iio.volread('data/20220929_background_Region1stack.tif')
plt.figure(figsize=(15, 15))
plt.imshow(stack_background[0])  # membrane
plt.show()
plt.imshow(stack_background[1])  # septin
plt.show()
plt.imshow(stack_background[2])  # actin
plt.show()

# split up into channels
septin_bg = stack_background[0]
actin_bg = stack_background[1]
membrane_bg = stack_background[2]

# numpy arrays?
septinbg_np = np.asarray(septin_bg)
actinbg_np = np.asarray(actin_bg)
membranebg_np = np.asarray(membrane_bg)

septin_background_avg = np.mean(septinbg_np)
actin_background_avg = np.mean(actinbg_np)
membrane_background_avg = np.mean(membranebg_np)

# subtract these values from the intensities spitted out by disguvery

# figure_angularpr, axes_angularpr = plt.subplots()
# plt.plot(results_angular_profiles_np[0:int(360/stepsize_theta-1),2], results_angular_profiles_np[0:int(360/stepsize_theta-1),3], color='magenta', label='septin')
# plt.plot(results_angular_profiles_np[int(360/stepsize_theta): int(360*2/stepsize_theta-1),2], results_angular_profiles_np[int(360/stepsize_theta): int(360*2/stepsize_theta-1),3], color='yellow', label='actin')
# plt.plot(results_angular_profiles_np[int(2*360/stepsize_theta): int(360*3/stepsize_theta-1),2], results_angular_profiles_np[int(360*2/stepsize_theta): int(360*3/stepsize_theta-1),3], color='cyan', label='lipids')
# plt.legend()
# axes_angularpr.set_xlabel('angle')
# axes_angularpr.set_ylabel('mean intensity')
# plt.show()

#%%

# figure_circles, axes_circles = plt.subplots()
# plt.scatter(detected_vesicles_np[:,1],detected_vesicles_np[:,2],c='red')
# for i in range(len(detected_vesicles_np)):
#     plt.annotate(i+1, (detected_vesicles_np[i,1], detected_vesicles_np[i,2]), c='red')
#     Drawing_uncolored_circle = plt.Circle((detected_vesicles_np[i,1], detected_vesicles_np[i,2]), detected_vesicles_np[i,3], fill=False, color='red')
#
# #axes_circles.set_aspect(1)
# #axes_circles.add_artist(Drawing_uncolored_circle)
#
# plt.imshow(membrane_np, origin='upper', cmap=matplotlib.colors.LinearSegmentedColormap.from_list('magentamap', ["black", "cyan"]))
# axes_circles.add_patch(Drawing_uncolored_circle)
# plt.show()





# #create a 512x512 black image
# img=np.zeros((512,512,3),np.uint8)
# #non filled circle
# img1 = cv2.circle(img,(256,256),63, (0,255,0), 8)
# #filled circle
# img1 = cv2.circle(img,(256,256),63, (0,0,255), -1)
# #now use a frame to show it just as displaying a image
# cv2.imshow("Circle",img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #%%
# imcircle = cv2.circle(membrane_np, (305, 308), 114, (0,255,0),8)
#
# cv2.imshow("Circle",imcircle)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # make plots with disguvery results
#
#
# #%%
#
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def func3(x, y):
#     return (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2 + y**2))
#
#
# # make these smaller to increase the resolution
# dx, dy = 0.05, 0.05
#
# x = np.arange(-3.0, 3.0, dx)
# y = np.arange(-3.0, 3.0, dy)
# X, Y = np.meshgrid(x, y)
#
# # when layering multiple images, the images need to have the same
# # extent.  This does not mean they need to have the same shape, but
# # they both need to render to the same coordinate system determined by
# # xmin, xmax, ymin, ymax.  Note if you use different interpolations
# # for the images their apparent extent could be different due to
# # interpolation edge effects
#
# extent = np.min(x), np.max(x), np.min(y), np.max(y)
# fig = plt.figure(frameon=False)
#
# Z1 = np.add.outer(range(8), range(8)) % 2  # chessboard
# im1 = plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest',
#                  extent=extent)
#
# Z2 = func3(X, Y)
# plt.show()
#
# im2 = plt.imshow(Z2, cmap=plt.cm.viridis,  interpolation='bilinear',
#                  extent=extent)
#
# plt.show()