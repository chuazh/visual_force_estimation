#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:20:30 2020

@author: charm

hyperparameter search
"""


import dataset_lib as dat
import models as mdl
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as opt
import numpy as np

if __name__ == "__main__":
    
    file_dir = '../experiment_data' # define the file directory for dataset
    
    model_type = "S"
    feat_extract = True
    force_align = False
    
    weight_file =  weight_file = "best_modelweights_" + model_type
    
    if model_type!="S" and feat_extract:
        weight_file="best_modelweights_" + model_type + "_ft"
        
    if force_align and model_type!= "V" :
        weight_file = weight_file + "_faligned"
        
    weight_file = weight_file + ".dat"
    
    # Define a transformation for the images
    trans_function = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    
    # We have to define the crop area of interest for the images
    # I hope to create cross-hairs in the gui so I can "aim" better during data collection.
    # That would help make the crop area consistent.
    
    crop_list = []
    
    for i in range(1,24):
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
    
    # set the logarithmic learning rate 
    learning_rates = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
    best_loss = np.inf
    best_lr = 0
    
    for lr in learning_rates:
        print('lr:{}'.format(lr))
        # define model
        if model_type == "VS":
            model = mdl.StateVisionModel(30, 54, 3,feature_extract=feat_extract)
        elif model_type == "S":
            model  = mdl.StateModel(54, 3)
        # create loss function
        criterion = nn.MSELoss(reduction='sum')
    
        # define optimization method
        optimizer = opt.Adam(model.parameters(),lr=lr)
        model,train_history,val_history,val_loss = mdl.train_model(model,
                                                             criterion, optimizer,
                                                             dataloaders, dataset_sizes,  
                                                             num_epochs=50,
                                                             model_type= model_type,
                                                             weight_file=weight_file,suppress_log=False, hyperparam_search=True)
        
        if val_loss<best_loss:
            print('found better lr: {}, loss:{}'.format(lr,val_loss))
            print('previous best lr:{}, loss:{}'.format(best_lr,best_loss))
            best_loss = val_loss
            best_lr = lr
            
    print("Learning Rate search completed, best LR:{}".format(best_lr))    
    #%%
    weight_decay_list = [0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
    best_loss = np.inf
    best_reg = 0
    
    for reg in weight_decay_list:
        print('reg:{}'.format(reg))
        # define model
        if model_type == "VS":
            model = mdl.StateVisionModel(30, 54, 3,feature_extract=feat_extract)
        elif model_type == "S":
            model  = mdl.StateModel(54, 3)
        # create loss function
        criterion = nn.MSELoss(reduction='sum')
    
        # define optimization method
        optimizer = opt.Adam(model.parameters(),lr=best_lr,weight_decay=reg) # if we want L2 loss set reg here
        model,train_history,val_history,val_loss = mdl.train_model(model,
                                                             criterion, optimizer,
                                                             dataloaders, dataset_sizes,  
                                                             num_epochs=50,
                                                             model_type= model_type,
                                                             weight_file=weight_file,
                                                             suppress_log=False,
                                                             L1_loss=0, # for L1 loss set it here
                                                             hyperparam_search=True)
        
        if val_loss<best_loss:
            print('found better reg: {}, loss:{}'.format(reg,val_loss))
            print('previous best reg:{}, loss:{}'.format(best_reg,best_loss))
            best_loss = val_loss
            best_reg = reg
            
    print("L2 Reg search completed, best L2 decay :{}".format(best_reg))   
    
    