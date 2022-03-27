# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:10:14 2021

@author: ritik
"""

data_root='D:/RADAR/CRUW'
annotations = 'D:/RADAR/CRUW/Expts/CFAR/'
from cruw import CRUW
import numpy as np
from scipy import signal
from scipy import ndimage

from cruw.mapping import ra2idx, idx2ra
import math
import json

# to Display the GT annotation
from cruw.visualization.utils import generate_colors_rgb
from cruw.visualization.draw_rf import draw_centers

from cruw.visualization.draw_rf import magnitude

import os
import matplotlib.pyplot as plt

from skimage.filters import gaussian,threshold_otsu,threshold_multiotsu, threshold_local
from skimage.feature import peak_local_max

dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')

def get_paths(seq_name, frame_id):
    image_path = os.path.join(data_root, 'sequences', 'train', seq_name, 
                              dataset.sensor_cfg.camera_cfg['image_folder'], 
                              '%010d.jpg' % frame_id)
    chirp_path = os.path.join(data_root, 'sequences', 'train', seq_name, 
                              dataset.sensor_cfg.radar_cfg['chirp_folder'],
                              '%06d_0000.npy' % frame_id)
    anno_path = os.path.join(data_root, 'annotations', 'train', seq_name + '.txt')
    return image_path, chirp_path, anno_path


def disp_annotation(chirp_path):    
    chirp = np.load(chirp_path)
    with open(anno_path, 'r') as f:
        lines = f.readlines()
    center_ids = []
    categories = []
    for line in lines:
        fid, rng, azm, class_name = line.rstrip().split()
        fid = int(fid)
        if fid == frame_id:
            rng = float(rng)
            azm = float(azm)
            rid, aid = ra2idx(rng, azm, dataset.range_grid, dataset.angle_grid)
            center_ids.append([rid, aid])
            categories.append(class_name)
       
    center_ids = np.array(center_ids)
    n_obj = len(categories)
    axs.set_title('GT Annotation')
    # ax.axis('off')
    colors = generate_colors_rgb(n_obj)
    draw_centers(axs, chirp, center_ids, colors, texts=categories)
    return(center_ids)

def get_annotation(chirp_path):
    chirp = np.load(chirp_path)
    with open(anno_path, 'r') as f:
        lines = f.readlines()
    center_ids = []
    categories = []
    for line in lines:
        fid, rng, azm, class_name = line.rstrip().split()
        fid = int(fid)
        if fid == frame_id:
            rng = float(rng)
            azm = float(azm)
            rid, aid = ra2idx(rng, azm, dataset.range_grid, dataset.angle_grid)
            center_ids.append([rid, aid])
            categories.append(class_name)
       
    return (center_ids, categories)


def Otsu(chirp_abs):
    #Computing Otsu threshold + Regional max
    g = gaussian(chirp_abs,sigma=1) #Gaussian smooth
    ot = threshold_otsu(g) #Otsu threshold
    g = g>ot
    xy = peak_local_max(g*chirp_abs, min_distance=2,threshold_abs=0.2)
    print(xy)
    axs.set_title('CFAR regional max')
    plt.imshow(g*chirp_abs,vmin=0, vmax=1, origin='lower')
    plt.autoscale(False)
    plt.plot(xy[:, 1],xy[:, 0], 'ro')
    return(xy)
    # print(xy)


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
    xy = peak_local_max(g*chirp_abs, min_distance=2,threshold_abs=0.2) #regional max
    axs.set_title('Re-Otsu regional Max')
    plt.imshow(g*chirp_abs,vmin=0, vmax=1, origin='lower')
    plt.autoscale(False)
    plt.plot(xy[:, 1],xy[:, 0], 'ro')
    return(xy)


def cfar(chirp_abs,mode='ca'):
    # import signal
    ra_size = chirp_abs.shape
    threshold = 9.633  # for a pfa of 1e-4
    win_param=[10,9,6,5]
    # win_param=[7,6,4,2]
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
    
    # Convert range-Azimuth map values to power
    ra_matrix = np.abs(chirp_abs) ** 2
    
    # Perform detection
    if(mode == 'ca'):
        print('ca')
        ra_windowed_sum = signal.convolve2d(ra_matrix, mask, mode='same')
        ra_avg_noise_power = ra_windowed_sum / (num_valid_cells_in_window)
        ra_snr = ra_matrix / (ra_avg_noise_power)
    
    elif(mode == 'os'):
        print('os')
        ra_snr = ra_matrix / os_cfar(ra_matrix, mask, num_valid_cells_in_window)
        
    
    hit_matrix = ra_snr > threshold
    
    #Finding regional Max
    xy = peak_local_max(hit_matrix*chirp_abs, min_distance=2,threshold_abs=0.2)
    print(xy)
    if(mode == 'ca'):
        axs.set_title('CA-CFAR regional max')
    elif(mode == 'os'):
        axs.set_title('OS-CFAR regional max')
    plt.imshow(hit_matrix*chirp_abs,vmin=0, vmax=1, origin='lower')
    plt.autoscale(False)
    plt.plot(xy[:, 1],xy[:, 0], 'ro')
    return(xy)
    
def os_cfar_old(ra_matrix, mask, num_valid_cells_in_window):
    kths = np.zeros_like(ra_matrix)
    x,y = int((mask.shape[0]-1)/2),int((mask.shape[1]-1)/2)
    ra_padded = np.pad(ra_matrix,[(x, ), (y, )],mode='constant' )
    sub_matrices = np.lib.stride_tricks.sliding_window_view(ra_padded, mask.shape)
    m = []
    for sub_m in sub_matrices:
        m.append(np.multiply(mask,sub_m))
    for i in range(128):
        for j in range(128):
            sorted_subm = sorted(np.array(m[i][j]).flatten())
            sorted_subm = [m for m in sorted_subm if m != [0]]
            kths[i][j] = sorted_subm[int(0.7*num_valid_cells_in_window[i][j]-1)]
    return(kths)

def os_cfar(ra_matrix, mask, num_valid_cells_in_window):
    rank = int(0.7*(mask == 1).sum())
    kths = ndimage.rank_filter(ra_matrix, rank, footprint=mask) #, mode='constant' )
    return(kths)
    
def adaptive(chirp_abs):
    g = gaussian(chirp_abs,sigma=1) #Gaussian smooth
    g = g>threshold_local(chirp_abs, 21)
    xy = peak_local_max(g*chirp_abs, min_distance=2,threshold_abs=0.2) #regional max
    axs.set_title('Adaptive regional max')
    plt.imshow(g*chirp_abs,vmin=0, vmax=1, origin='lower')
    plt.autoscale(False)
    plt.plot(xy[:, 1],xy[:, 0], 'ro')
    return(xy)
    # print(xy)

from PIL import Image, ImageDraw
def generate_annotated_image(chirp_abs, sigma=(2,2), r1=2, r2=1, r3=1):
    """
    generate the density map by creating gaussian at the center of each dot annotation
        1. draw ellipses at the coordinates of targets
        2. apply a gaussian filter on the image

    Parameters
    ----------
    chirp_abs : np.ndarray
        Range Azimuth input image.
    sigma : tuple, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence. The default is (2,2).
    r1 : int
        Radius of the dot annotation for car
    r2 : int
        Radius of the dot annotation for cyclist
    r3 : int
        Radius of the dot annotation for pedestrian
    

    Returns
    -------
    Annotated density map to be used with FCRN.

    """
    
    image = Image.fromarray(chirp_abs) # array to PIL Image object
    draw = ImageDraw.Draw(image) #Creates an object that can be used to draw in the given image
    
    target_ids,categories = get_annotation(chirp_path)
    
    # for each target location id, draw a circle at the id of appropriate radius
    for i in range(len(target_ids)):
        id = target_ids[i]
        
        if (categories[i]=='car'):
            draw.ellipse([id[1]-r1, id[0]-r1, id[1]+r1, id[0]+r1], fill = 'red', outline ='red')
        
        if (categories[i]=='pedestrian'):
            draw.ellipse([id[1]-r2, id[0]-r2, id[1]+r2, id[0]+r2], fill = 'red', outline ='red')
        
        if (categories[i]=='cyclist'):
            draw.ellipse([id[1]-r3, id[0]-r3, id[1]+r3, id[0]+r3], fill = 'red', outline ='red')
    
    #Apply gaussian filter over the dot annotated image
    image = ndimage.gaussian_filter(image, sigma, order=0)
    return(image)
    

seq_name = '2019_04_30_MLMS000' # CFAR catching otsu misses
frame_id = 145
data = {'2019_04_09_BMS1000': '896', '2019_04_09_BMS1001': '896','2019_04_09_BMS1002': '896', '2019_04_09_CMS1002': '896', '2019_04_09_PMS1000': '897', '2019_04_09_PMS1001': '897', '2019_04_09_PMS2000': '897', '2019_04_09_PMS3000': '897', '2019_04_30_MLMS000': '899', '2019_04_30_MLMS001': '899', '2019_04_30_MLMS002': '899', '2019_04_30_PBMS002': '899', '2019_04_30_PBMS003': '899', '2019_04_30_PCMS001': '899', '2019_04_30_PM2S003': '899', '2019_04_30_PM2S004': '899', '2019_05_09_BM1S008': '899', '2019_05_09_CM1S004': '899', '2019_05_09_MLMS003': '899', '2019_05_09_PBMS004': '899', '2019_05_09_PCMS002': '899', '2019_05_23_PM1S012': '897', '2019_05_23_PM1S013': '897', '2019_05_23_PM1S014': '897', '2019_05_23_PM1S015': '897', '2019_05_23_PM2S011': '897', '2019_05_29_BCMS000': '899', '2019_05_29_BM1S016': '899', '2019_05_29_BM1S017': '899', '2019_05_29_MLMS006': '899', '2019_05_29_PBMS007': '899', '2019_05_29_PCMS005': '899', '2019_05_29_PM2S015': '899', '2019_05_29_PM3S000': '899', '2019_09_29_ONRD001': '1697', '2019_09_29_ONRD002': '1678', '2019_09_29_ONRD005': '1696', '2019_09_29_ONRD006': '1662', '2019_09_29_ONRD011': '1681', '2019_09_29_ONRD013': '1689'}




seq_name = '2019_04_30_MLMS002'
frame_id = 115

image_path, chirp_path, anno_path = get_paths(seq_name, frame_id)
    
        #Computing magnitude of RF
chirp_sample = np.load(chirp_path)
chirp_abs = magnitude(chirp_sample, 'RISEP')

anno_img = generate_annotated_image(chirp_abs)
plt.imshow(np.asarray(anno_img), origin = 'lower')





















"""
from PIL import Image, ImageDraw
image = Image.fromarray(chirp_abs)
draw = ImageDraw.Draw(image)

target_ids,categories = get_annotation(chirp_path)

for i in range(len(target_ids)):
    id = target_ids[i]
    
    if (categories[i]=='car'):
        draw.ellipse([id[0]-2, id[1]-2, id[0]+2, id[1]+2], fill = 'red', outline ='red')
    
    if (categories[i]=='pedestrian'):
        draw.ellipse([id[0]-1, id[1]-1, id[0]+1, id[1]+1], fill = 'red', outline ='red')
    
    if (categories[i]=='cyclist'):
        draw.ellipse([id[0]-1, id[1]-1, id[0]+1, id[1]+1], fill = 'red', outline ='red')

image = ndimage.gaussian_filter(image, sigma=(1, 1), order=0)
plt.imshow(np.asarray(image))

from PIL import Image, ImageDraw
image = Image.fromarray(chirp_abs)
draw = ImageDraw.Draw(image)
annotated = draw.point((100, 100), 'red')

image2 = Image.fromarray(chirp_abs)
draw2 = ImageDraw.Draw(image2)
draw2.ellipse([98, 98, 101, 101], fill = 'red', outline ='red')
draw2.ellipse([22, 22, 25, 25], fill = 'red', outline ='red')

# image2.show()
plt.imshow(np.asarray(image2),vmin=0, vmax=1)


numpydata = np.asarray(image)


image.show()

plt.imshow(chirp_abs)   

diff = numpydata - chirp_abs
new = chirp_abs + diff
plt.imshow(new)


"""


"""
fig = plt.figure()
fig.set_size_inches(16, 5)
        
        
axs = plt.subplot(161)
axs.set_title('Input Radar image')
plt.imshow(chirp_abs,vmin=0, vmax=1, origin='lower')
plt.xlabel('Azimuth')
plt.ylabel('Range')
        
axs = plt.subplot(162)
disp_annotation(chirp_path)
        
axs = plt.subplot(163)
# print(Otsu(chirp_abs))
# print(re_Otsu(chirp_abs))
print(re_Otsu(chirp_abs))
        
axs = plt.subplot(164)
cfar(chirp_abs,'ca')

axs = plt.subplot(165)
cfar(chirp_abs,'os')

axs = plt.subplot(166)
adaptive(chirp_abs)

"""
# plt.savefig("D:\RADAR\CRUW\Expts\plots\output.jpg")



# seqs = ['2019_05_29_BCMS000','2019_05_29_MLMS006','2019_04_30_MLMS002','2019_04_30_MLMS000','2019_04_30_PCMS001','2019_04_30_MLMS001']

# p = "D:\RADAR\CRUW\Expts\plots"
# for seq_name in seqs:
#     path = os.path.join(p,seq_name)
#     os.mkdir(path)
#     n_frames = int(data[seq_name])
#     for frame_id in range(n_frames+1):
#         image_path, chirp_path, anno_path = get_paths(seq_name, frame_id)
    
#         #Computing magnitude of RF
#         chirp_sample = np.load(chirp_path)
#         chirp_abs = magnitude(chirp_sample, 'RISEP')
        
        
#         fig = plt.figure()
#         fig.set_size_inches(16, 5)
        
        
#         axs = plt.subplot(141)
#         axs.set_title('Input Radar image')
#         plt.imshow(chirp_abs,vmin=0, vmax=1, origin='lower')
#         plt.xlabel('Azimuth')
#         plt.ylabel('Range')
        
#         axs = plt.subplot(142)
#         disp_annotation(chirp_path)
        
#         axs = plt.subplot(143)
#         print(Otsu(chirp_abs))
        
#         axs = plt.subplot(144)
#         cfar(chirp_abs)
        
#         plt.savefig(os.path.join(path,str(frame_id)))
        
        
        
"""
seqs = ['2019_04_30_MLMS002']

p = "D:\RADAR\CRUW\Expts\plots2"
for seq_name in seqs:
    path = os.path.join(p,seq_name)
    os.mkdir(path)
    n_frames = int(data[seq_name])
    for frame_id in range(n_frames+1):
        image_path, chirp_path, anno_path = get_paths(seq_name, frame_id)
    
        #Computing magnitude of RF
        chirp_sample = np.load(chirp_path)
        chirp_abs = magnitude(chirp_sample, 'RISEP')
        
        
        fig = plt.figure()
        fig.set_size_inches(16, 5)
        
        
        axs = plt.subplot(162)
        disp_annotation(chirp_path)
        
        axs = plt.subplot(163)
        # print(Otsu(chirp_abs))
        # print(re_Otsu(chirp_abs))
        print(re_Otsu(chirp_abs))
                
        axs = plt.subplot(164)
        cfar(chirp_abs,'ca')
        
        axs = plt.subplot(165)
        cfar(chirp_abs,'os')
        
        axs = plt.subplot(166)
        adaptive(chirp_abs)
        
        plt.savefig(os.path.join(path,str(frame_id)))

"""