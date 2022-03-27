# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:23:07 2021

@author: ritik
"""

data_root='D:/RADAR/CRUW'
annotations = 'D:/RADAR/CRUW/Expts/CFAR/'
from cruw import CRUW
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.spatial import distance
from skimage.feature import match_template
from cruw.mapping import ra2idx, idx2ra

import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix

from cruw.visualization.draw_rf import magnitude
import os

from skimage.filters import gaussian,threshold_otsu
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
    return(center_ids,categories)

def re_Otsu(chirp_abs):
    denoised = gaussian(chirp_abs,sigma=1) #Gaussian smooth
    otsu_th = threshold_otsu(denoised) #Otsu threshold
    g = denoised>otsu_th
    labeled_image, num_objects = ndimage.label(g)
    for i in range(num_objects):
        area = ndimage.sum(g, labeled_image, index=[i+1])
        if area>200:
            loc=ndimage.find_objects(labeled_image)[i]
            section = denoised[loc]>threshold_otsu(denoised[loc])
            g[loc] = section
    xy = peak_local_max(g*chirp_abs, min_distance=2,threshold_abs=0.2) #regional max
    return(xy,g)

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
        # print('ca')
        ra_windowed_sum = signal.convolve2d(ra_matrix, mask, mode='same')
        ra_avg_noise_power = ra_windowed_sum / (num_valid_cells_in_window)
        ra_snr = ra_matrix / (ra_avg_noise_power)
    
    elif(mode == 'os'):
        # print('os')
        ra_snr = ra_matrix / os_cfar(ra_matrix, mask, num_valid_cells_in_window)
        
    
    hit_matrix = ra_snr > threshold
    
    #Finding regional Max
    xy = peak_local_max(hit_matrix*chirp_abs, min_distance=2)
    return(xy, hit_matrix)
    

def os_cfar(ra_matrix, mask, num_valid_cells_in_window):
    rank = int(0.7*(mask == 1).sum())
    kths = ndimage.rank_filter(ra_matrix, rank, footprint=mask) #, mode='constant' )
    return(kths)

def conf_matrix(coors_truth,coors_pred,max_dist=4):
    """
    - match each predicted target coordinate with ground truth actual coordinate
    
    - identify ground truth coordinates which were never detected
    
    - return confusion matrix for the predicted coordinates and groundtruth 
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
    dict
        key: predicted target coordinate ; Value: closest actual target coordinate
        and
        key: (0,) ; Value: Actual target coordinate which was not identified in prediction
    
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

    return({y:x for x,y in coor_map.items()},[[len(TP),len(FN)],[len(FP),100]])

def rad_displacement(objs, template_coor, chirp_abs_prev_frame, n, img_dim = (128,128), ps = 7):
    """
    return the displacement between the location of object in current frame 
    and previous frame
        
        -- match the template with prev frame's image patch centered at
            coordinates from which template is taken from
        -- find peak coordinates
        -- find distance between the current target coordinates and matched template coordinates
        -- return distance 

    Parameters
    ----------
    objs : list
        slices of the detected objects.
    template_coor : numpy.ndarray
        range and azimuth coordinate.
    chirp_abs_prev_frame : numpy.ndarray
        Range - Azimuth image of the previous frame.
    n : int
        index - object label.
    ps : int
        value to determine the patch size cropped around target

    Returns
    -------
    rad_disp : int
        Displacement between the location of object in current frame and previorus frame.

    """
    # segment a patch in the prev frame that 
    patch = np.zeros(img_dim)
    patch[max(0,objs[n-1][0].start - ps): min(img_dim[0],objs[n-1][0].stop + ps),
          max(0,objs[n-1][1].start - ps): min(img_dim[1],objs[n-1][1].stop + ps)] = 1
    
    matched_cc = match_template(chirp_abs_prev_frame*patch, chirp_abs[objs[n-1]])
    disp_coor = np.column_stack(np.where(matched_cc==np.amax(matched_cc)))
    rad_disp = np.linalg.norm(template_coor-disp_coor)

    return rad_disp

def compute_features(xy, g, chirp_abs, chirp_abs_prev_frame, truth_coor, classes_loc):
    """
    - Compute the features for predicted targets 
    - Append the features into 'features' list
        
    Features - Azimuth, Range, area, object lenght, width, mean ref power, 
                    variance of ref power, maximum ref power, radial velocity

    Parameters
    ----------
    xy : Array of int64
        List of predicted target coordinates.
    g : Array of Bool
        Detected Object Hit matrix.
        Dimentsion - 128x128
    chirp_abs : numpy.ndarray
        Range - Azimuth image.
    truth_coor : Array of int64
        True target coordinates for the image.
    classes_loc : dict
        True target coordinates and target class as key - value pairs.

    Returns
    -------
    None.

    """
    #label each object in segmented image and count them
    labeled_image, num_objects = ndimage.label(g*chirp_abs)
    
    #find the object slices in the labelled array
    objs = ndimage.find_objects(labeled_image)    
    
    #
    coor_map, conf_mat = conf_matrix(truth_coor, xy)

    for xy_ in xy:
        for ind in range(1,num_objects+1):
            if(xy_[0]>=objs[ind-1][0].start and xy_[0]<=objs[ind-1][0].stop and 
               xy_[1]>=objs[ind-1][1].start and xy_[1]<=objs[ind-1][1].stop):
                
                area = ndimage.sum(g, labeled_image, index=[ind])
                spread_len = objs[ind-1][1].stop - objs[ind-1][1].start
                spread_height = objs[ind-1][0].stop - objs[ind-1][0].start
                mean = ndimage.mean(g*chirp_abs, labeled_image, index=[ind])
                variance = ndimage.variance(g*chirp_abs, labeled_image, index=[ind])
                maximum = ndimage.maximum(g*chirp_abs, labeled_image, index=[ind])
                rad_disp = rad_displacement(objs, xy_, chirp_abs_prev_frame, ind)
                
                if(tuple(xy_) in coor_map.keys()):
                    target_class = classes_loc[coor_map[tuple(xy_)]]
                else:
                    target_class = 'not Object'
                
                features4.append([xy_[0], xy_[1], area[0], spread_len, spread_height,
                                 mean[0], variance[0], maximum[0], rad_disp, target_class])
    return
    



seqs = ['2019_05_29_BCMS000' ,'2019_05_29_MLMS006','2019_04_30_MLMS002','2019_04_30_MLMS000','2019_04_30_PCMS001','2019_04_30_MLMS001']
data = {'2019_04_09_BMS1000': '896','2019_04_09_BMS1001': '896','2019_04_09_BMS1002': '896', '2019_04_09_CMS1002': '896', '2019_04_09_PMS1000': '897', '2019_04_09_PMS1001': '897', '2019_04_09_PMS2000': '897', '2019_04_09_PMS3000': '897', '2019_04_30_MLMS000': '899', '2019_04_30_MLMS001': '899', '2019_04_30_MLMS002': '899', '2019_04_30_PBMS002': '899', '2019_04_30_PBMS003': '899', '2019_04_30_PCMS001': '899', '2019_04_30_PM2S003': '899', '2019_04_30_PM2S004': '899', '2019_05_09_BM1S008': '899', '2019_05_09_CM1S004': '899', '2019_05_09_MLMS003': '899', '2019_05_09_PBMS004': '899', '2019_05_09_PCMS002': '899', '2019_05_23_PM1S012': '897', '2019_05_23_PM1S013': '897', '2019_05_23_PM1S014': '897', '2019_05_23_PM1S015': '897', '2019_05_23_PM2S011': '897', '2019_05_29_BCMS000': '899', '2019_05_29_BM1S016': '899', '2019_05_29_BM1S017': '899', '2019_05_29_MLMS006': '899', '2019_05_29_PBMS007': '899', '2019_05_29_PCMS005': '899', '2019_05_29_PM2S015': '899', '2019_05_29_PM3S000': '899', '2019_09_29_ONRD001': '1697', '2019_09_29_ONRD002': '1678', '2019_09_29_ONRD005': '1696', '2019_09_29_ONRD006': '1662', '2019_09_29_ONRD011': '1681', '2019_09_29_ONRD013': '1689'}

features = [] # list to append the computed features into


for seq_name in seqs:
    n_frames = int(data[seq_name])
    for frame_id in range(1,n_frames+1):
        image_path, chirp_path, anno_path = get_paths(seq_name, frame_id)
        i2, prev_chrip_path, a2 = get_paths(seq_name, frame_id-1)
        
        #Computing magnitude of RF
        chirp_sample = np.load(chirp_path)
        chirp_abs = magnitude(chirp_sample, 'RISEP')
        
        #Computing magnitude of RF for the previous frame
        chirp_abs_prev_frame = magnitude(np.load(prev_chrip_path),'RISEP')
        
        # get the target coordinates-xy and segmented image - seg_image using Otsu
        # xy, seg_image = re_Otsu(chirp_abs)
        # xy, seg_image = cfar(chirp_abs,'os')
        xy, seg_image = cfar(chirp_abs)
        
        #get the ground truth coordinates of target location and their classes
        truth_coor, classes = disp_annotation(chirp_path)
        
        #create a dictionary to hold target location and corresponding class as key-value pairs
        classes_loc = dict()
        for i in range(len(classes)):
            classes_loc[tuple(truth_coor[i])] = classes[i]
        
        compute_features(xy, seg_image, chirp_abs, chirp_abs_prev_frame, truth_coor, classes_loc)
    print('done')

#Create a dataframe of the measured features
features_df = pd.DataFrame(features, columns = ['Azimuth', 'Range','area',
                                                'Length','Height','mean','variance',
                                                'maximum','radial disp','class'])


"""
encoding classes as numbers
0 - not an object
1 - cyclist
2 - pedestrian
3 - car
"""
temp = features_df['class'].map({'not Object':0, 'cyclist':1, 'pedestrian':1, 'car':1})
features_df['class'] = temp

##Storing column values
columns_list = list(features_df.columns)

##Separting ip names from data
features_= list(set(columns_list)-set(['class']))

##Storing ground truth in y
y = features_df['class'].values
x = features_df[features_].values

###Train test split
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=0)


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print(prediction)
conf=confusion_matrix(test_y,prediction)
print(conf)


accuracy = accuracy_score(test_y, prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

"""
HYPER-PARAMETER TUNING WITH RANDOMIZED SEARCH
RandomizedSearchCV() 
    - Perform Randomized search on hyper parameters.
    - A fixed number of parameter settings is sampled from the specified distributions.
    
    Learning_rate: (default = 0.3)
     	Makes the model more robust by shrinking the weights on each step

    n_estimators (default = 100)
        Number of gradient boosted trees. Equivalent to number of boosting rounds.

    Subsample (default = 1)
     	Denotes the fraction of observations to be randomly samples for each tree.

    colsample_bytree (default=1)
        Denotes the fraction of columns to be randomly sampled for each tree.

    max_depth (default=6)
        The maximum depth of a tree.
        Used to control over-fitting as higher depth will allow model to learn \
            relations very specific to a particular sample.

    
    scoring:
        Strategy to evaluate the performance of the cross-validated model on the test set.
"""

params = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.8),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.5),
              'min_child_weight': [1, 2, 3, 4]
              }
#f1 scorer
scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average = 'weighted')

#f2 scorer
scorer_f2 = sklearn.metrics.make_scorer(sklearn.metrics.fbeta_score, beta = 2.0, average = 'micro')

clf = RandomizedSearchCV(model,
                          param_distributions=params,
                          scoring=scorer_f2,
                          n_iter=5,
                          verbose=1)
clf.fit(train_x, train_y)
print("Best parameters:", clf.best_params_)

"""
Best parameters: {'colsample_bytree': 0.5862645459646291, 
                  'learning_rate': 0.5045100089955603, 
                  'max_depth': 8, 'min_child_weight': 1, 
                  'n_estimators': 948, 
                  'subsample': 0.6445663800126393} """


model_after_tuning = XGBClassifier(colsample_bytree = clf.best_params_['colsample_bytree'],
                      learning_rate = clf.best_params_['learning_rate'], 
                      max_depth= clf.best_params_['max_depth'],
                      min_child_weight = clf.best_params_['min_child_weight'], 
                      n_estimators = clf.best_params_['n_estimators'],
                      subsample = clf.best_params_['subsample'])

model_after_tuning.fit(train_x,train_y)
prediction=model_after_tuning.predict(test_x)
print(prediction)
conf=confusion_matrix(test_y,prediction)
print(conf)
accuracy = accuracy_score(test_y, prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

importance = model.feature_importances_

from matplotlib import pyplot

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

## print classification report
print(sklearn.metrics.classification_report(test_y,prediction))



#pickle the features list
import pickle
with open("features.txt", "wb") as fp:   #Pickling
   pickle.dump(features, fp)
        
        
        
#############################################################################

""" binary classification using xgboost"""

