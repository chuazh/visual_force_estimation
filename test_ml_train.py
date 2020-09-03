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
    
    file_dir = '../ML dvrk 080320'
    
    trans_function = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    crop_list = []
    
    for i in range(1,14):
        crop_list.append((57,320,462,462))
    for i in range(14,17):
        crop_list.append((40,400,462,462))
    for i in range(17,20):
        crop_list.append((57,180,462,462))
    for i in range(20,23):
        crop_list.append((70,145,462,462))
    crop_list.append((30,250,462,462))
    
    train_set = dat.ImgDataset(file_dir,file_dir,
                               data_sets=[1,3,5],
                               transform = trans_function,
                               crop_list=crop_list,
                               include_torque=False,
                               custom_state=None)
    
    val_set = dat.ImgDataset(file_dir,file_dir,
                           data_sets=[2,4],
                           transform = trans_function,
                           crop_list=crop_list,
                           include_torque=False,
                           custom_state=None)
    
    #generate_grid(train_set,64)
    
    train_loader = data.DataLoader(train_set,batch_size=16,shuffle=True)
    val_loader = data.DataLoader(val_set,batch_size=16,shuffle=False)
    dataloaders = {'train':train_loader,'val':val_loader}
    dataset_sizes ={'train':len(train_set),'val':len(val_set)}
    
    # define model
    vs_model = mdl.StateVisionModel(30, 54, 3,feature_extract=True)
    # create loss function
    criterion = nn.MSELoss(reduction='sum')
    # define optimization method
    optimizer = opt.Adam(vs_model.parameters())
    vs_model,train_history,val_history = mdl.train_model(vs_model,
                                                         criterion, optimizer,
                                                         dataloaders, dataset_sizes,
                                                         num_epochs=3,
                                                         model_type="VS",
                                                         weight_file="best_modelweights_ft.dat")
    
    