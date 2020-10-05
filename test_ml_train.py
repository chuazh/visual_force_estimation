#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:11:52 2020

@author: charm
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

def generate_grid(dataset,num_imgs=64,orig=True):
    
    dataloader = data.DataLoader(dataset,batch_size=num_imgs,shuffle=True)

    # Get a batch of training data
    inputs, aug_inputs ,labels = next(iter(dataloader))
    
    if orig:
        for i in range(inputs.shape[0]):
            inputs[i,:,:,:] = unnorm(inputs[i,:,:,:])
    
    # Make a grid from batch
    
    out = torchvision.utils.make_grid(inputs)
    imshow(out)
    
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)
    fig,ax = plt.subplots(figsize = (10,10))
    ax.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

unnorm = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

if __name__ == "__main__":
    
    file_dir = '../experiment_data' # define the file directory for dataset
    
    model_type = "V"
    feat_extract = False
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
    
    for i in range(1,39):
        #crop_list.append((50,350,300,300))
        crop_list.append((270-150,480-150,300,300))
        
    train_list = [1,3,5,7,
                  8,10,12,14,
                  15,17,19,21]
    val_list = [2,6,
                9,13,
                16,20]
    #test_list = [4,11,18,
                 #22,23,24,25,26,27,28,29,32,33]
    test_list = [9,13]
    config_dict={'file_dir':file_dir,
                 'include_torque': False,
                 'spatial_forces': force_align,
                 'custom_state': None,
                 'batch_size': 32,
                 'crop_list': crop_list,
                 'trans_function': trans_function}
    
    dataloaders,dataset_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict,augment=True)
    #%%
    #generate_grid(dataloaders['train'].dataset,64)

    # define model
    if model_type == "VS":
        model = mdl.StateVisionModel(30, 54, 3,feature_extract=feat_extract)
    elif model_type == "S":
        model  = mdl.StateModel(54, 3)
    elif model_type == "V":
        model = mdl.VisionModel(3)
    
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
                                                         weight_file=weight_file,
                                                         suppress_log=False)
    
    