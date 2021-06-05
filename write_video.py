#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:44:30 2020

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

#data = np.loadtxt("labels.txt",delimiter=",")
dataset_list=[32]

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 5}

matplotlib.rc('font', **font)    

#%% 
#test_list_full =  [4,11,18,22,23,24,25,26,27,28,29,32,33,34,36,37,38,39]
test_list_full =  [4,11,18,22,23,24,25,26,27,28,29,32,33,34,36,37,38,39]
for dataset_num in test_list_full:
    dataset_idx = test_list_full.index(dataset_num)
    
    model_types = ["V","S","VS"]
    color_dict = {"V":(14,127,255),"S":(44,160,44),"VS":(40,39,214)}
    # load the predictions
    pred_dict = {}
    for model_type in model_types:
        pred_dict[model_type] = pickle.load(open("../../clean_code/preds_"+model_type+"_test.preds","rb"))
    # get the reference forces
    labels = np.loadtxt('../labels_'+str(dataset_num)+'.txt',delimiter=",")
    force = labels[:,1:4]
    time = labels[:,0]
    
    # glob the images 
    heatmap_VS = glob.glob("imageset_captum_VS_"+str(dataset_num)+"/*.jpg")
    heatmap_VS = humansorted(heatmap_VS)
    heatmap_V = glob.glob("imageset_captum_V_"+str(dataset_num)+"/*.jpg")
    heatmap_V = humansorted(heatmap_V)
    original = glob.glob("../imageset_"+str(dataset_num)+"/*.jpg")
    original = humansorted(original)
    
    out = cv2.VideoWriter('summary_video_'+str(dataset_num)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'),30,(672,972),True)
    with tqdm.tqdm(total=len(heatmap_VS),desc="dataset "+str(dataset_num)) as pbar:
        for frame_num in range(len(heatmap_VS)):
            #frame_num = np.random.randint(0,3000)
            # for the given frame:
            frame_force = force[frame_num,:]
            frame_pred = {}
            gap_between_image = 30
            for model_type in model_types:
                frame_pred[model_type] = pred_dict[model_type][dataset_idx][frame_num]
            frame_heatmap_VS = cv2.imread(heatmap_VS[frame_num])
            cv2.putText(frame_heatmap_VS,"X", (10,210), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0,0,0),2)
            cv2.putText(frame_heatmap_VS,"Y", (10+224,210), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0,0,0),2)
            cv2.putText(frame_heatmap_VS,"Z", (10+224+224,210), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0,0,0),2)
            frame_heatmap_V = cv2.imread(heatmap_V[frame_num])
            cv2.putText(frame_heatmap_V,"X", (10,210), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0,0,0),2)
            cv2.putText(frame_heatmap_V,"Y", (10+224,210), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0,0,0),2)
            cv2.putText(frame_heatmap_V,"Z", (10+224+224,210), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0,0,0),2)
            frame_heatmap_V = cv2.copyMakeBorder(frame_heatmap_V,gap_between_image,gap_between_image,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
            cv2.putText(frame_heatmap_V,"Vision Model GRADCAM", (10,gap_between_image-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0,0,0),2)
            cv2.putText(frame_heatmap_V,"Vision + State Model GRADCAM", (10,224+gap_between_image*2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0,0,0),2)
            # stack the frames
            frame = np.vstack((frame_heatmap_V,frame_heatmap_VS))
            
            # set up graphing area params
            line_length = 100
            left = int(112-line_length/2)
            image_height = 224
            gap_from_image_to_plots = 30
            box_max_height = 100 # scale to 10 newtons
            v_pos = 224*2+gap_between_image*2+gap_from_image_to_plots+box_max_height
            box_width = 30
            
            # make the frame larger
            enlarged_frame = cv2.copyMakeBorder(frame,0,gap_from_image_to_plots+2*box_max_height+10,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
            
            cv2.line(enlarged_frame,(left,v_pos),(left+line_length,v_pos),(0,0,0),2)
            cv2.putText(enlarged_frame,"Forces:", (10,v_pos-box_max_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0),2)
            for axis in range(3):
                cv2.putText(enlarged_frame,"Ref", (left,v_pos-box_max_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0),2)
                cv2.rectangle(enlarged_frame,(left,int(v_pos-frame_force[axis]*10)),(left+box_width,v_pos),(180,119,31),-1)
                for n,(model_type) in enumerate(model_types,1):
                    cv2.putText(enlarged_frame,model_type, (left+box_width*n+5,v_pos-box_max_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0),2)
                    cv2.rectangle(enlarged_frame,(left+box_width*n,int(v_pos-frame_pred[model_type][axis]*10)),(left+box_width*(n+1),v_pos),color_dict[model_type],-1)
                left = left+image_height
            
            # for each axis we plot the relevant forces up to some time window.
            # if the sampling rate is 30hz then 3 seconds is 90 frames:
            plt.close('all')
            plt.ioff()
            if frame_num < 90:
                fig,ax = plt.subplots(1,3,sharey=True,figsize=(3,1),dpi=224)
                for axis in range(3):
                    ax[axis].plot(time[:frame_num],force[:frame_num,axis],linewidth=0.5)
                    for model_type in model_types:
                        ax[axis].plot(time[:frame_num],pred_dict[model_type][dataset_idx][:frame_num,axis],linewidth=0.5)
                        ax[axis].set_ylim((-10,10))
            else:
                fig,ax = plt.subplots(1,3,sharey=True,figsize=(3,1),dpi=224)
                for axis in range(3):
                    ax[axis].plot(time[frame_num-90:frame_num],force[frame_num-90:frame_num,axis],linewidth=0.5)
                    for model_type in model_types:
                        ax[axis].plot(time[frame_num-90:frame_num],pred_dict[model_type][dataset_idx][frame_num-90:frame_num,axis],linewidth=0.5)
                        ax[axis].set_ylim((-10,10))
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            img_fig = cv2.cvtColor(np.asarray(buf)[:,:,:-1],cv2.COLOR_RGB2BGR)
            enlarged_frame = np.vstack((enlarged_frame,img_fig))
            
            #frame_original = cv2.resize(cv2.imread(original[frame_num])[120:420,330:630,:],(972,972))
            #enlarged_frame = np.hstack((enlarged_frame,frame_original))
            plt.ion() 
            out.write(enlarged_frame)
            pbar.update(1)
            #plt.figure()    
            #plt.imshow(cv2.cvtColor(enlarged_frame,cv2.COLOR_BGR2RGB))
        
        out.release()

#%%

for dataset in dataset_list:

    list_img_files = glob.glob("imageset_captum_" +str(dataset) + '/*.jpg') # collect all the jpg files
    list_img_files = humansorted(list_img_files)
    
    #list_img = [cv2.imread(file_img) for file_img in list_img_files] # read them all in
    #cv2.imshow('image',img)
    
    # convert to video:
    out = cv2.VideoWriter('captum_video_'+str(dataset)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'),30,(672,224),True)
    frame_count = 0
    
    with tqdm.tqdm(total=len(list_img_files)) as pbar:
        for file in list_img_files:
            frame = cv2.imread(file)
            '''force = data[frame_count,0:4]
            cv2.rectangle(frame,(10,15),(100,15+20),(0,0,0),2)
            cv2.rectangle(frame,(55,15+2),(int(55+45*force[1]/10),15+20-2),(0,255,0),-1)
            cv2.rectangle(frame,(10,45),(100,45+20),(0,0,0),2)
            cv2.rectangle(frame,(55,45+2),(int(55+45*force[2]/10),45+20-2),(0,255,0),-1)
            cv2.rectangle(frame,(10,75),(100,75+20),(0,0,0),2)
            cv2.rectangle(frame,(55,75),(int(55+45*force[3]/10),75+20-2),(0,255,0),-1)
            cv2.putText(frame,str(round(data[frame_count,0],2)),(55,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)'''
            out.write(frame)
            pbar.update(1)
            frame_count += 1
    out.release()
    
#%% Simple output video clips

# set the list of datasets

test_list_full =  [4,11,18,
                    22,23,
                    24,25,
                    26,27,
                    28,29,
                    32,33,
                    34,36,
                    37,38,39]
 
 
condition_list = ['center','right','left',
                   'right_less','right_less',
                   'right_more','right_more',
                   'left_less','left_less',
                   'left_more','left_more',
                   'new_tool','new_tool',
                   'new_material','new_material',
                   'center','right','left']

    
dataset_list = [4,11,18]
start_idx = 800
out = cv2.VideoWriter('../experiment_data/plain_video_'+str(dataset)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'),30,(300,300),True)

for dataset in dataset_list:
    list_img_files = glob.glob("../experiment_data/imageset_" +str(dataset) + '/*.jpg') # collect all the jpg files
    list_img_files = humansorted(list_img_files)
    selected_files = list_img_files[start_idx:start_idx+300+1]
    with tqdm.tqdm(total=len(selected_files)) as pbar:
        for file in selected_files:
            frame = cv2.imread(file)
            cropped_frame = frame[270-150:270+150,480-150:480+150,:]
            out.write(cropped_frame)
            pbar.update(1)
    out.release()
    
#%% stacked grid

# first pull all the required datasets into a list of list
dataset_list = [46,42,47]
#dataset_list = [32,34]    
list_of_lists = []
start_idx = 750
frame_size = 150
out = cv2.VideoWriter('../experiment_data/grid_video_z.avi', cv2.VideoWriter_fourcc(*'MJPG'),30,(frame_size*len(dataset_list),frame_size),True)
for dataset in dataset_list:
    list_img_files = glob.glob("../experiment_data/imageset_" +str(dataset) + '/*.jpg') # collect all the jpg files
    list_img_files = humansorted(list_img_files)
    selected_files = list_img_files[start_idx:start_idx+300+1]
    list_of_lists.append(selected_files)

# now iterate through frame
length_file = len(selected_files)

with tqdm.tqdm(total=length_file) as pbar:
    for idx in range(length_file):
        first_idx = 1
        for files in list_of_lists:
            if first_idx==1:
                frame = cv2.imread(files[idx])[270-150:270+150,480-150:480+150,:]
                frame = cv2.resize(frame,(frame_size,frame_size))
            else:
                new_frame = cv2.imread(files[idx])[270-150:270+150,480-150:480+150,:]
                new_frame = cv2.resize(new_frame,(frame_size,frame_size))
                frame = np.concatenate((frame,new_frame),axis=1)
            first_idx += 1
        out.write(frame)
        pbar.update(1)
out.release()

#%% vanilla with force prediction

test_list_full =  [4,11,18,
                    22,23,
                    24,25,
                    26,27,
                    28,29,
                    32,33,
                    34,36,
                    37,38,39,41,42]

dataset = 28
start_idx = 400#81
frame_size = 300
list_img_files = glob.glob("../../../experiment_data/imageset_" +str(dataset) + '/*.jpg') # collect all the jpg files
list_img_files = humansorted(list_img_files)[80:-10]

dataset_idx = test_list_full.index(dataset)

model_types = ["V","S","VS","RNN","D"]
color_dict = {"V":(0,147/255,44/255),"S":(243/255,28/255,13/255),"VS":(0/255,79/255,212/255),"D":(202/255,169/255,0/255),"RNN":(202/255,105/255,255/255)}
line_dict = {"V":1,"S":1,"VS":1,"D":0.5,"RNN":0.5}

# get the reference forces
labels = np.loadtxt('../../../experiment_data/labels_'+str(dataset)+'.txt',delimiter=",")
# load the predictions
pred_dict = {}
pred_dict['V'] = pickle.load(open("preds_V.preds","rb"))[dataset_idx][80:-10,:]
pred_dict['VS'] = pickle.load(open("preds_VS.preds","rb"))[dataset_idx][80:-10,:]
pred_dict['S'] = pickle.load(open("preds_S.preds","rb"))[dataset_idx][80:-10,:]

pred_dict['RNN'] = pickle.load(open("../preds_V_RNN.preds","rb"))[dataset_idx]
pad_length =  pred_dict['RNN'].shape[0] - (labels.shape[0]-80)
pred_dict['RNN'] = pred_dict['RNN'][:-pad_length-10,:3]

pred_dict['D'] = pickle.load(open('dynamic_model_preds_test_031421.dat','rb'), encoding='latin1')['pred_data'][dataset_idx][70:,:]

# trim the labels
labels = labels[80:-10,:]
force = labels[:,1:4]
time = labels[:,0]
lag = 40

# creat video object
out = cv2.VideoWriter('vanilla_traj_'+str(dataset)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'),30,(frame_size*2,frame_size*4+5),True)

# start drawing each frame
with tqdm.tqdm(total=30*15) as pbar:
    for frame_num in range(30*15):
        frame_img = cv2.imread(list_img_files[start_idx+frame_num]) # get the frame
        frame_img = frame_img[270-150:270+150,480-150:480+150,:] #crop it
        #enlarge the frame
        enlarged_frame = cv2.copyMakeBorder(frame_img,0,5,200,100,cv2.BORDER_CONSTANT,value=[255,255,255])
        # draw the plot
        plt.close('all')
        plt.ioff()
        fig,ax = plt.subplots(3,1,sharex=True,sharey=False,figsize=(2,3),dpi=300)
        for axis in range(3):
            ax[axis].plot(time[start_idx+frame_num-lag:start_idx+frame_num+lag],force[start_idx+frame_num-lag:start_idx+frame_num+lag,axis],linewidth=1.5,color="black")
            for model_type in model_types:
                ax[axis].plot(time[start_idx+frame_num-lag:start_idx+frame_num+lag],pred_dict[model_type][start_idx+frame_num-lag:start_idx+frame_num+lag,axis],linewidth=line_dict[model_type],color=color_dict[model_type])
                ax[axis].plot(time[start_idx+frame_num],pred_dict[model_type][start_idx+frame_num,axis],'o',markersize=3,color=color_dict[model_type])
                if axis == 2:
                    ax[axis].set(xlabel="time [s]",ylabel="Z force [N]")
                elif axis == 1:
                    ax[axis].set(ylabel="Y force [N]")
                else:
                    ax[axis].set(ylabel="X force [N]")
                ax[axis].set_ylim((-5,5))
                #ax[axis].set_aspect(0.1)
        #fig.tight_layout()
        plt.subplots_adjust(left=0.2,bottom=0.1,right=0.975,top=1)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_fig = cv2.cvtColor(np.asarray(buf)[:,:,:-1],cv2.COLOR_RGB2BGR)
        enlarged_frame = np.vstack((enlarged_frame,img_fig))
        out.write(enlarged_frame)
        pbar.update(1)

out.release()

#%% GRADCAM spaced out

dataset = 34
start_idx = 200
out = cv2.VideoWriter('031421_experiments/heatmaps/gradcam_VS_'+str(dataset)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'),30,(682,224),True)
list_img_files = glob.glob("031421_experiments/heatmaps/imageset_captumVS_" +str(dataset) + '/*.jpg') # collect all the jpg files
list_img_files = humansorted(list_img_files)

with tqdm.tqdm(total=30*15) as pbar:
    for frame_num in range(30*15):
        frame_img = cv2.imread(list_img_files[start_idx+frame_num])
        out.write(frame_img)
out.release()
