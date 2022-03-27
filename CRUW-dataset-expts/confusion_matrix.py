# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 18:52:15 2021

@author: ritik
"""

import numpy as np
from scipy.spatial import distance

def conf_matrix(coors_truth,coors_pred,max_dist=4):
    """
    return confusion matrix for the predicted coordinates and groundtruth 
    target coordinates

    Parameters
    ----------
    coors_truth : numpy.ndarray or list
        list of actual target coordinates.
    coors_pred : numpy.ndarray or list
        list of predicted target coordinates.
    max_dist : int, optional
        Maximum Euclidean distance between the predicted and actual coordinates
        to classify prediction as true positive. 
        The default is 4.

    Returns
    -------
    list
        Confusion matrix - [[True Pos, False Neg],[False Pos],[True Neg]]

    """
    TP = []
    FP = []
    FN = []
    coor_map = dict()
    for xy_truth in coors_truth:
        dist, c_ = 100, [0]
        for xy_pred in coors_pred:
            d_temp = distance.euclidean(xy_truth, xy_pred)
            if(d_temp<max_dist and d_temp<dist):
                dist = d_temp
                c_ = xy_pred
        coor_map[tuple(xy_truth)]=tuple(c_)
    
    for xy_pred in coors_pred:
        if (tuple(xy_pred) not in coor_map.values()):
            FP.append(tuple(xy_pred))
    for k,v in coor_map.items():
        if (v == (0,)):
            FN.append(k)
        else:
            TP.append(v)

    return([[len(TP),len(FN)],[len(FP),100]])
                   
        
                
                