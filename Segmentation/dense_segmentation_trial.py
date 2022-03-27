# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:10:59 2022

@author: ritik
"""

import numpy as np
import matplotlib.pyplot as plt
import thresholds

id_no = "000524"

ra_proc_1 = np.load("D:/RADAR/Carrada/2019-09-16-12-52-12/range_angle_processed/"+id_no+".npy")
plt.imshow(ra_proc_1)

ra_raw_1 = np.load("D:/RADAR/Carrada/2019-09-16-12-52-12/range_angle_raw/"+id_no+".npy")
plt.imshow(ra_raw_1)

ra_numpy_1 = np.load("D:/RADAR/Carrada/2019-09-16-12-52-12/range_angle_numpy/"+id_no+".npy")
plt.imshow(ra_numpy_1)

ra_proc_int = ra_proc_1.astype(int)
plt.imshow(ra_proc_int)

segmentation_anno_1 = np.load("D:/RADAR/Carrada/2019-09-16-12-52-12/annotations/dense/000524/range_angle.npy")
segmentation_anno_2 = np.load("D:/RADAR/Carrada/2020-02-28-12-16-05/annotations/dense/000313/range_angle.npy")
segmentation_anno_3 = np.load("D:/RADAR/Carrada/2020-02-28-12-16-05/annotations/dense/000046/range_angle.npy")

plt.imshow(segmentation_anno_1[0])
plt.imshow(segmentation_anno_1[3])
plt.imshow(segmentation_anno_2[0])
plt.imshow(segmentation_anno_2[1])
plt.imshow(segmentation_anno_3[0])
plt.imshow(segmentation_anno_3[2])

thresholds.Otsu(ra_proc_1)

from PIL import Image
import glob

for f_name in glob.glob("D:/RADAR/Carrada/2019-09-16-12-52-12/annotations/dense/*/range_angle.npy"):
    # print(f_name.split(.npy)[0][-6:])
    a = f_name.split('range_angle.npy')[0][-7:-1]
    dense_anno = np.logical_not(np.load(f_name)[0])
    im = Image.fromarray(dense_anno)
    im.save(f"D:/RADAR/Carrada_seg_anno/2019-09-16-12-52-12-{a}.png")
    # print(f"D:/RADAR/Carrada_seg_anno/2019-09-16-12-52-12-{a}.png")
