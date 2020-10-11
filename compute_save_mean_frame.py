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
