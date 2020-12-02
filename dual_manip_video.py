#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:50:09 2020

@author: charm
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import glob
import copy
import tqdm
from natsort import humansorted
import pickle

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

import copy
#%%

def draw_force_arrows(frame,pred1,pred2,summed_force):

    #scale the forces
    max_force = 10 #N
    bar_width = 30
    force_scaling = 100/max_force #pixels per newton
    scaled_force_1 = pred1*force_scaling
    scaled_force_2 = pred2*force_scaling
    scaled_sum = summed_force*force_scaling
    
    height_anchor = 300
    
    # unit vectors
    unit_vectors = np.array([[-1,0],[0.19106143, 0.98157808],[0.23162053, -0.97280621]])
    #y_vec = np.array([0.83915299,-0.54389545])
    #z_vec = np.array([0.23162053, -0.97280621])
    
    l_anchor = 300
    l_start = np.array([[l_anchor,height_anchor],[l_anchor,height_anchor],[l_anchor,height_anchor]],np.int32)
    l_end = copy.copy(l_start) + unit_vectors*np.expand_dims(scaled_force_1,1)
    l_end = l_end.astype(np.int32)

    
    r_anchor = 960-300
    r_start = np.array([[r_anchor,height_anchor],[r_anchor,height_anchor],[r_anchor,height_anchor]],np.int32)
    r_end = copy.copy(r_start) + unit_vectors*np.expand_dims(scaled_force_2,1)
    r_end = r_end.astype(np.int32)
    
    height_anchor = 100
    c_anchor = 480
    scaled_pred_sum = scaled_force_1+scaled_force_2
    c_start = np.array([[c_anchor,height_anchor],[c_anchor,height_anchor],[c_anchor,height_anchor]],np.int32)
    c_start2 = copy.copy(c_start) 
    c_end = copy.copy(c_start) + unit_vectors*np.expand_dims(scaled_sum,1)
    c_end2 = copy.copy(c_start) + unit_vectors*np.expand_dims(scaled_pred_sum,1)
    c_end = c_end.astype(np.int32)
    c_end2 = c_end2.astype(np.int32)
    
    
    # color tuples
    for i in range(3):
        # left manip
        if i == 0:
            color = (0,0,255)
            color2 = (0,0,127)
        elif i == 1:
            color = (0,255,0)
            color2 = (0,127,0)
        else:
            color = (255,0,0)
            color2 = (127,0,0)
        
        frame = cv2.arrowedLine(frame,tuple(l_start[i,:]),tuple(l_end[i,:]),color,4)
        frame = cv2.arrowedLine(frame,tuple(r_start[i,:]),tuple(r_end[i,:]),color,4)
        frame = cv2.arrowedLine(frame,tuple(c_start[i,:]),tuple(c_end[i,:]),color,4)
        frame = cv2.arrowedLine(frame,tuple(c_start2[i,:]),tuple(c_end2[i,:]),color2,4)
    
    return frame
    
def draw_force_bars(frame,pred1,pred2,summed_force):
    
    #scale the forces
    max_force = 10 #N
    bar_width = 30
    force_scaling = 100/max_force #pixels per newton
    scaled_force_1 = pred1*force_scaling
    scaled_force_2 = pred2*force_scaling
    scaled_sum = summed_force*force_scaling
    
    height_anchor = 300
    
    l_anchor = 300
    l_start = np.array([[l_anchor,height_anchor],[l_anchor,height_anchor+bar_width],[l_anchor,height_anchor+bar_width*2]],np.int32)
    l_end = copy.copy(l_start)
    l_end[:,1] = l_end[:,1] + np.array([bar_width,bar_width,bar_width])
    l_end[:,0] = l_end[:,0]+scaled_force_1.astype(np.int32)
    
    r_anchor = 960-300
    r_start = np.array([[r_anchor,height_anchor],[r_anchor,height_anchor+bar_width],[r_anchor,height_anchor+bar_width*2]],np.int32)
    r_end = copy.copy(r_start)
    r_end[:,1] = r_end[:,1] + np.array([bar_width,bar_width,bar_width])
    r_end[:,0] = r_end[:,0]+scaled_force_2.astype(np.int32)
    
    height_anchor = 10
    c_anchor = 480
    scaled_pred_sum = scaled_force_1+scaled_force_2
    c_start = np.array([[c_anchor,height_anchor],[c_anchor,height_anchor+bar_width],[c_anchor,height_anchor+bar_width*2]],np.int32)
    c_start2 = copy.copy(c_start)+ np.array([[0,5],[0,5],[0,5]])
    c_end = copy.copy(c_start)
    c_end2 = copy.copy(c_start)
    c_end[:,1] = c_end[:,1] + np.array([bar_width,bar_width,bar_width])
    c_end[:,0] = c_end[:,0] + scaled_pred_sum.astype(np.int32)
    c_end2[:,1] = c_end2[:,1] + np.array([bar_width-5,bar_width-5,bar_width-5])
    c_end2[:,0] = c_end2[:,0] + scaled_sum.astype(np.int32)
    
    # color tuples
    for i in range(3):
        # left manip
        if i == 0:
            color = (0,0,255)
            color2 = (0,0,127)
        elif i == 1:
            color = (0,255,0)
            color2 = (0,127,0)
        else:
            color = (255,0,0)
            color2 = (127,0,0)
        
        frame = cv2.rectangle(frame,tuple(l_start[i,:]),tuple(l_end[i,:]),color,-1)
        frame = cv2.rectangle(frame,tuple(r_start[i,:]),tuple(r_end[i,:]),color,-1)
        frame = cv2.rectangle(frame,tuple(c_start[i,:]),tuple(c_end[i,:]),color,-1)
        frame = cv2.rectangle(frame,tuple(c_start2[i,:]),tuple(c_end2[i,:]),color2,-1)
    
    return frame


#%%
    
for dataset in range(4):
    dataset += 1
    out = cv2.VideoWriter('../dual_manip_data/video_arrow_'+str(dataset)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'),30,(960,540),True)
    list_img_files = glob.glob("../dual_manip_data/imageset_" +str(dataset) + '/*.jpg') # collect all the jpg files
    list_img_files = humansorted(list_img_files)
    selected_files = list_img_files
    pred_PSM1 = np.loadtxt("dual_PSM1_"+str(dataset)+".preds")
    pred_PSM2 = np.loadtxt("dual_PSM2_"+str(dataset)+".preds")
    labels = np.loadtxt("../dual_manip_data/PSM1/labels_"+str(dataset)+".txt",delimiter=",")[:,1:4]
    with tqdm.tqdm(total=len(selected_files)) as pbar:
        for n,file in enumerate(selected_files,0):
            frame = cv2.imread(file)
            pred1 = pred_PSM1[n,:]
            pred2 = pred_PSM2[n,:]
            gt = labels[n,:]
            #frame = draw_force_bars(frame,pred1,pred2,gt)
            frame = draw_force_arrows(frame,pred1,pred2,gt)
            #cropped_frame = frame[270-150:270+150,480-150:480+150,:]
            out.write(frame)
            pbar.update(1)
    out.release()