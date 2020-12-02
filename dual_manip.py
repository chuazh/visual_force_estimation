#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:08:33 2020

@author: charm
"""


import dataset_lib as dat
import models as mdl
from torchvision import transforms
from torch.utils import data

import torch 
import numpy as np
import torch.nn as nn

import captum 
from captum.attr import GuidedBackprop

import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
import seaborn as sns

import pickle


#%% create PSM1 Data loader
dataset = 4

file_dir = '../PSM1_data' # define the file directory for dataset
model_type = "S"
train_list = [1,3,5,7]
val_list = [1]
test_list = [1]
config_dict={'file_dir':file_dir,
         'include_torque': False,
         'spatial_forces': False,
         'custom_state': None,
         'batch_size': 32,
         'crop_list':None,
         'trans_function': None}

loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)

# from the loader_dict extract the mean and std of the train set
mean1 = loader_dict["train"].dataset.mean
std1 = loader_dict["train"].dataset.stdev

#load the PSM1 dual manip data
file_dir = '../dual_manip_data/PSM1' # define the file directory for dataset
train_list = [1]
val_list = [1]
test_list = [dataset]

config_dict={'file_dir':file_dir,
         'include_torque': False,
         'spatial_forces': False,
         'custom_state': None,
         'batch_size': 32,
         'crop_list':None,
         'trans_function': None}

loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)
test_loader = loader_dict["test"]
test_loader.dataset.mean = mean1
test_loader.dataset.stdev =std1
test_loader.dataset.label_array = test_loader.dataset.raw_label_array
test_loader.dataset.normalize_state(mean1,std1)

model  = mdl.StateModel(54, 3)
model.load_state_dict(torch.load("best_modelweights_S_PSM1.dat"))


predictions1 = mdl.evaluate_model(model,test_loader,model_type = model_type)

np.savetxt("dual_PSM1_1.pred",predictions1)

#%% Do this for PSM2

file_dir = '../experiment_data' # define the file directory for dataset
model_type = "S"
train_list = [1,3,5,7,
                  8,10,12,14,
                  15,17,19,21]
val_list = [1]
test_list = [1]
config_dict={'file_dir':file_dir,
         'include_torque': False,
         'spatial_forces': False,
         'custom_state': None,
         'batch_size': 32,
         'crop_list':None,
         'trans_function': None}

loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)

# from the loader_dict extract the mean and std of the train set
mean2 = loader_dict["train"].dataset.mean
std2 = loader_dict["train"].dataset.stdev

#load the PSM1 dual manip data
file_dir = '../dual_manip_data/PSM2' # define the file directory for dataset
train_list = [1]
val_list = [1]
test_list = [dataset]

config_dict={'file_dir':file_dir,
         'include_torque': False,
         'spatial_forces': False,
         'custom_state': None,
         'batch_size': 32,
         'crop_list':None,
         'trans_function': None}

loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)
test_loader = loader_dict["test"]
test_loader.dataset.mean = mean2
test_loader.dataset.stdev = std2
test_loader.dataset.label_array = test_loader.dataset.raw_label_array
test_loader.dataset.normalize_state(mean2,std2)

model  = mdl.StateModel(54, 3)
model.load_state_dict(torch.load("best_modelweights_S.dat"))

predictions2 = mdl.evaluate_model(model,test_loader,model_type = model_type)

#%%
pred1 = (predictions1*std1[1:4])+mean1[1:4]
pred2 = (predictions2*std2[1:4])+mean2[1:4]
np.savetxt("dual_PSM1_"+str(dataset)+".preds",pred1)
np.savetxt("dual_PSM2_"+str(dataset)+".preds",pred2)
resultant_force = pred1+pred2
label_force = test_loader.dataset.raw_label_array[:,1:4]

fig, ax = plt.subplots(3,2,sharex=True)

time = np.arange(0,resultant_force.shape[0]/30,1/30)[:pred1.shape[0]]

ax[0,0].plot(time,label_force[:,0])
ax[0,0].set_ylabel("X Force [N]")
ax[0,0].plot(time,resultant_force[:,0],linewidth=1)
ax[0,1].plot(time,resultant_force[:,0],linewidth=0.7)
ax[0,1].plot(time,pred1[:,0],linewidth=1)
ax[0,1].plot(time,pred2[:,0],linewidth=1)
ax[0,0].set_ylim(-5,5)

ax[1,0].plot(time,label_force[:,1])
ax[1,0].plot(time,resultant_force[:,1],linewidth=1)
ax[1,1].plot(time,resultant_force[:,1],linewidth=0.7)
ax[1,1].plot(time,pred1[:,1],linewidth=1)
ax[1,1].plot(time,pred2[:,1],linewidth=1)
ax[1,0].set_ylabel("Y Force [N]")
ax[1,0].set_ylim(-5,5)
    
ax[2,0].plot(time,label_force[:,2])
ax[2,0].plot(time,resultant_force[:,2],linewidth=1)
ax[2,1].plot(time,resultant_force[:,2],linewidth=0.7)
ax[2,1].plot(time,pred1[:,2],linewidth=1)
ax[2,1].plot(time,pred2[:,2],linewidth=1)
ax[2,0].set_ylabel("Z Force [N]")
ax[2,0].set_xlabel("Time")
ax[2,1].set_xlabel("Time")
ax[2,0].set_ylim(-10,5)