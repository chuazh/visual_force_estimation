#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:02:35 2020

@author: charm
"""

import models as mdl
import model_eval
import dataset_lib as dat
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as transforms
import seaborn as sns
import pandas as pd
import pickle

qty = ['t','fx','fy','fz','tx','ty','tz',
       'px','py','pz','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz',
       'q1','q2','q3','q4','q5','q6','q7',
       'vq1','vq2','vq3','vq4','vq5','vq6','vq7',
       'tq1','tq2','tq3','tq4','tq5','tq6','tq7',
       'q1d','q2d','q3d','q4d','q5d','q6d','q7d',
       'tq1d','tq2d','tq3d','tq4d','tq5d','tq6d','tq7d',
       'psm_fx','psm_fy','psm_fz','psm_tx','psm_ty','psm_tz',
       'J1','J2','J3','J4','J5','J6','J1','J2','J3','J4','J5','J6',
       'J1','J2','J3','J4','J5','J6','J1','J2','J3','J4','J5','J6',
       'J1','J2','J3','J4','J5','J6','J1','J2','J3','J4','J5','J6']
       

if __name__ == "__main__":
    
    file_dir = '../experiment_data' # define the file directory for dataset
    
    model_type = "S"
    feat_extract = False
    force_align = False
    
    weight_file =  weight_file = "best_modelweights_" + model_type
    
    if model_type!="S" and feat_extract:
        weight_file="best_modelweights_" + model_type + "_ft"
        
    if force_align and model_type!= "V" :
        weight_file = weight_file + "_faligned"
    
    pretrained_file = weight_file +".dat"
   
    
    # Define a transformation for the images
    trans_function = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    
    # We have to define the crop area of interest for the images
    # I hope to create cross-hairs in the gui so I can "aim" better during data collection.
    # That would help make the crop area consistent.
    
    crop_list = []
    
    for i in range(1,34):
        #crop_list.append((50,350,300,300))
        crop_list.append((270-150,480-150,300,300))
        
    train_list = [1,3,5,7,
                  8,10,12,14,
                  15,17,19,21]
    val_list = [2,6,
                9,13,
                16,20]
    test_list = [4,11,18,
                 22,23,24,25,26,27,28,29,32,33]
    config_dict={'file_dir':file_dir,
                 'include_torque': False,
                 'spatial_forces': force_align,
                 'custom_state': None,
                 'batch_size': 32,
                 'crop_list': crop_list,
                 'trans_function': trans_function}
    
    dataloaders,dataset_sizes = dat.init_dataset(train_list,val_list,val_list,model_type,config_dict)
    
    #%% First ablation: remove the force data
    
    force_features = ['tq1','tq2','tq3','tq4','tq5','tq6','tq7',
       'q1d','q2d','q3d','q4d','q5d','q6d','q7d',
       'tq1d','tq2d','tq3d','tq4d','tq5d','tq6d','tq7d',
       'psm_fx','psm_fy','psm_fz','psm_tx','psm_ty','psm_tz']
    
    pos_features = ['px','py','pz','qx','qy','qz','qw',
       'vx','vy','vz','wx','wy','wz',
       'q1','q2','q3','q4','q5','q6','q7',
       'vq1','vq2','vq3','vq4','vq5','vq6','vq7',
       'q1d','q2d','q3d','q4d','q5d','q6d','q7d']
    
    for ab_cond in ['F','P']:
        if ab_cond == 'F':
            mask_feature = force_features
            ab_weight_file = weight_file + "_F.dat"
        else:
            mask_feature = pos_features
            ab_weight_file = weight_file + "_P.dat"
            
        mask = np.isin(qty,mask_feature,invert=False)
        #mask = np.isin(qty,force_features,invert=False)
        
        for loader in dataloaders.values():
            loader.dataset.mask_labels(mask)
        
        if model_type == "VS":
            model = mdl.StateVisionModel(30, 54, 3,feature_extract=feat_extract)
        elif model_type == "S":
            model  = mdl.StateModel(54, 3)
        
        pretrained_weights = torch.load(pretrained_file)
        #model.load_state_dict(pretrained_weights)
        # create loss function
        criterion = nn.MSELoss(reduction='sum')
        # define optimization method
        optimizer = opt.Adam(model.parameters(),lr=0.001,weight_decay=0)
        model,train_history,val_history,_ = mdl.train_model(model,
                                                             criterion, optimizer,
                                                             dataloaders, dataset_sizes,  
                                                             num_epochs=100,
                                                             L1_loss=0.001,
                                                             model_type= model_type,
                                                             weight_file=ab_weight_file,
                                                             suppress_log=False)
    