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

def generate_grid(dataset,num_imgs=64):
    
    dataloader = data.DataLoader(dataset,batch_size=num_imgs,shuffle=True)

    # Get a batch of training data
    inputs, aug_inputs ,labels = next(iter(dataloader))
    
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


if __name__ == "__main__":
    
    file_dir = '../ML dvrk 081320' # define the file directory for dataset
    
    model_type = "S"
    feat_extract = True
    
    if model_type!="S" and feat_extract:
        weight_file="best_modelweights_" + model_type + "_ft.dat"
    else:
        weight_file = "best_modelweights_" + model_type + ".dat"
    
    # Define a transformation for the images
    trans_function = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    
    # We have to define the crop area of interest for the images
    # I hope to create cross-hairs in the gui so I can "aim" better during data collection.
    # That would help make the crop area consistent.
    
    crop_list = []
    
    for i in range(1,8):
        crop_list.append((50,350,300,300))
        
    '''
    for i in range(1,14):
        crop_list.append((57,320,462,462))
    for i in range(14,17):
        crop_list.append((40,400,462,462))
    for i in range(17,20):
        crop_list.append((57,180,462,462))
    for i in range(20,23):
        crop_list.append((70,145,462,462))
    crop_list.append((30,250,462,462))
    '''
    train_list = [1,2,3,5,6]
    val_list = [4,7]
    config_dict={'file_dir':file_dir,
                 'include_torque': False,
                 'custom_state': None,
                 'batch_size': 128,
                 'crop_list': crop_list,
                 'trans_function': trans_function}
    
    dataloaders,dataset_sizes = dat.init_dataset(train_list,val_list,val_list,model_type,config_dict)
    
    #generate_grid(train_set,64)

    # define model
    if model_type == "VS":
        model = mdl.StateVisionModel(30, 54, 3,feature_extract=feat_extract)
    elif model_type == "S":
        model  = mdl.StateModel(54, 3)
    # create loss function
    criterion = nn.MSELoss(reduction='sum')
    # define optimization method
    optimizer = opt.Adam(model.parameters(),lr=0.01)
    model,train_history,val_history = mdl.train_model(model,
                                                         criterion, optimizer,
                                                         dataloaders, dataset_sizes,
                                                         num_epochs=50,
                                                         model_type= model_type,
                                                         weight_file=weight_file,no_pbar=False)
    
    