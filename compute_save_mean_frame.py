#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 21:43:16 2020

@author: charm
"""
import copy
import PIL.Image as im
from natsort import humansorted
import glob
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

def mean_frame_computation_exp_smoothing(current_frame, mean_frame, alpha=0.5):

        """
        Mean frame computation based on exponential smoothing:

            s(t) = x(t);                                    t = 0
            s(t) = alpha * x(t) + ( 1 - alpha ) * s(t-1);   t > 1, 0 < alpha < 1

        Note: This function was used for real-time computation of the mean-frame.

        :param current_frame: Current frame.
        :param alpha: Smoothing factor, 0 < alpha < 1.
        :param initial_frame:
                If True:
                    s(t) = x(t).
                Otherwise:
                    s(t) = alpha * x(t) + ( 1 - alpha ) * s(t-1).
        :return: Mean frame.
        """


        mean_frame_new = alpha * current_frame + (1 - alpha) * mean_frame
        
        return mean_frame_new

#%% MEAN FRAME COMPUTATION
# We calculate the mean frames for each video set and save it in the directory over which it belongs.
for sequence in range(39):
    file_dir = "../experiment_data/imageset_"+str(sequence+1)
    file_list = glob.glob(file_dir+ "/*.jpg")
    file_list = humansorted(file_list)
    it = 0
    mean_frame = 0.0
    
    with tqdm.tqdm(total=len(file_list)) as pbar:
        for file in file_list:
            x = cv2.imread(file).astype(np.float)
            if it == 0:
                mean_frame = x
            else:
                #mean_frame = mean_frame_computation_exp_smoothing(x, mean_frame,alpha=0.075)
                mean_frame =  mean_frame + x 
            it += 1
            pbar.update(1)
    
    mean_frame = mean_frame.astype(np.float)/len(file_list)
    cv2.imwrite(file_dir+"/image_mean.jpg",mean_frame)
    
normalized_frame = (x.astype(np.uint8)-mean_frame.astype(np.uint8)).astype(np.uint8)+127
#plt.imshow(mean_frame.astype(np.uint8))
#plt.imshow()

#%% Save images

import dataset_lib as dat
import torchvision.transforms as transform
import os
# initialize the datasets

crop_list = []
for i in range(0,39):
    #crop_list.append((50,350,300,300))
    crop_list.append((270-150,480-150,300,300))
model_type = "V_RNN"
trans_function = transform.Lambda(lambda x: x)
file_dir = "../experiment_data"
config_dict={'file_dir':file_dir,
                 'include_torque': False,
                 'spatial_forces': False,
                 'custom_state': None,
                 'batch_size': 1,
                 'crop_list': crop_list,
                 'trans_function': trans_function}
train_list = [1]
val_list = [1]

try:
    os.system("mkdir "+file_dir+"/space_time")
except:
    print("directory exists")

for dataset_num in range(0,1):
    test_list = [dataset_num+1]
    dataloaders,dataset_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict,augment=False)

    test_loader = dataloaders['test']
    dataset_dir =  file_dir + "/space_time/imageset_" + str(dataset_num+1)
    mkdir_command = "mkdir " + dataset_dir
    
    try:
        os.system(mkdir_command)
        print("making " + mkdir_command)
    except:
        print("directory exists")
        
    with tqdm.tqdm(total=len(test_loader)) as pbar:
        for img_num,(x,z,y) in enumerate(test_loader,1):
            x = (x.squeeze().cpu().numpy().transpose(1,2,0) *255).astype(np.uint8)
            imagesave = dataset_dir + "/img_" +str(img_num)+".jpg"
            cv2.imwrite(imagesave,x)
            pbar.update(1)
