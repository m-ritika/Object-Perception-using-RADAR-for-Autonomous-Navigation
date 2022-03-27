# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:45:30 2021

@author: ritik
"""
import numpy as np
from scipy import signal
from skimage.filters import gaussian,threshold_otsu
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from scipy import ndimage

def Otsu(chirp_abs):
    #Computing Otsu threshold + Regional max
    g = gaussian(chirp_abs,sigma=1) #Gaussian smooth
    ot = threshold_otsu(g) #Otsu threshold
    g = g>ot
    xy = peak_local_max(g*chirp_abs, min_distance=5,threshold_abs=0.2) #regional max
    plt.imshow(g*chirp_abs,vmin=0, vmax=1, origin='lower')
    plt.title('Otsu regional Max')
    plt.autoscale(False)
    plt.plot(xy[:, 1],xy[:, 0], 'ro')
    return(xy)

def re_Otsu(chirp_abs):
    denoised = gaussian(chirp_abs,sigma=1) #Gaussian smooth
    otsu_th = threshold_otsu(denoised) #Otsu threshold
    g = denoised>otsu_th
    labeled_image, num_objects = ndimage.label(g)
    objs = ndimage.find_objects(labeled_image)
    for i in range(num_objects):
        area = ndimage.sum(g, labeled_image, index=[i+1])
        if area>200:
            loc=ndimage.find_objects(labeled_image)[i]
            section = denoised[loc]>threshold_otsu(denoised[loc])
            g[loc] = section
    xy = peak_local_max(g*chirp_abs, min_distance=5,threshold_abs=0.2) #regional max
    plt.imshow(g*chirp_abs,vmin=0, vmax=1, origin='lower')
    plt.autoscale(False)
    plt.plot(xy[:, 1],xy[:, 0], 'ro')
    return(xy)

def cfar(chirp_abs):
    # import signal
    ra_size = chirp_abs.shape
    threshold = 9.6428  # for a pfa of 1e-5
    win_param=[7,6,4,2]
    win_width = win_param[0] # number of training cells in azimuth dim
    win_height = win_param[1] # number of training cells in range dim
    guard_width = win_param[2] # number of guard cells in azimuth dim
    guard_height = win_param[3] # number of guard cells in range dim
    
    # Create window mask with guard cells
    mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
    mask[win_height - guard_height:win_height + 1 + guard_height, win_width - guard_width:win_width + 1 + guard_width] = 0
    
    #threshold value - PFA
    threshold = 10 ** (threshold / 10)
    
    # Number cells within window around CUT; used for averaging operation.
    num_valid_cells_in_window = signal.convolve2d(np.ones(ra_size, dtype=float), mask, mode='same')
    print(num_valid_cells_in_window)
    
    # Convert range-Azimuth map values to power
    ra_matrix = np.abs(chirp_abs) ** 2
    
    # Perform detection
    ra_windowed_sum = signal.convolve2d(ra_matrix, mask, mode='same')
    ra_avg_noise_power = ra_windowed_sum / (num_valid_cells_in_window)
    ra_snr = ra_matrix / (ra_avg_noise_power)
    hit_matrix = ra_snr > threshold
    
    #Finding regional Max
    xy = peak_local_max(hit_matrix*chirp_abs, min_distance=5,threshold_abs=0)
    plt.imshow(hit_matrix*chirp_abs,vmin=0, vmax=1, origin='lower')
    plt.title('CFAR regional max')
    plt.autoscale(False)
    plt.plot(xy[:, 1],xy[:, 0], 'ro')
    

