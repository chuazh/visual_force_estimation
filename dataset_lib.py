#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:15:11 2020

@author: charm

Visual Force Estimation Library
"""

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import models,transforms
import PIL.Image as im
import numpy as np
import glob
import natsort

class ImgDataset(data.Dataset):
    
    '''
    Characterizes a dataset for Img Dataset for PyTorch
    
    label_dir: the directory where the label files are located
    image_dir: the directory where the image files organized as folders are located
    data_sets: as list of datasets corresponding with the image folders to be used to construct the dataset
    transform: the transform function to apply to the images
    crop_list: a list of crop specs as 4-element tuples (top,left,height,width)
    
    For the directory globbing to work properly, the image directories must be numerically ordered with no missing
    sets. For example folders need to be imageset_1, imageset_2, imageset_3. They cannot be imageset_1, imageset_3, imageset_4.
    Doing the latter will result in specifying [2] in data_sets to give imageset_3 instead of imageset_2.
    
    The an iterator call on the class generates an example.
    The example is a 3-element tuple (img,state,label)
    
    
    '''
    def __init__(self, label_dir, image_dir, data_sets = None, transform=None, crop_list = [], include_torque = False, custom_state = None):
        '''
        Initialization
           exclude_index is a list denoting which datasets to exclude indexed
           from 1
        '''
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.transform = transform
        self.crop_list = crop_list
        self.include_torque = include_torque
        self.custom_state = custom_state
        self.image_folder_prefix = '/imageset*'
        
        if len(self.crop_list)>0:
            assert(len(self.crop_list)==len(glob.glob(image_dir+self.image_folder_prefix)))
        
        self.label_array = self.read_labels(self.label_dir,data_sets)
        self.image_list,self.dataset_list = self.generate_image_list(self.image_dir,data_sets)
        
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_array)

    def __getitem__(self, index):
        'Generates one sample of data'
        x = im.open(self.image_list[index])
        
        
        dataset_idx = self.dataset_list[index]-1
        if len(self.crop_list)> 0 :
            '''use the crop list'''
            t = self.crop_list[dataset_idx][0]
            l = self.crop_list[dataset_idx][1]
            h = self.crop_list[dataset_idx][2]
            w = self.crop_list[dataset_idx][3]
            x = transforms.functional.crop(x,top=t,left=l,height=h,width=w)
        else:
            '''Use the default crop'''
            #x = transforms.functional.crop(x,top=57,left=320,height=462,width=462)
            x = transforms.functional.crop(x,top=100,left=320,height=250,width=250) #new dataset
        
        '''
        if self.dataset_list[index] in [14,15,16]:
          x = transforms.functional.crop(x,top=40,left=400,height=462,width=462) # for dataset 14-16
        elif self.dataset_list[index] in [17,18,19]:
          x = transforms.functional.crop(x,top=57,left=180,height=462,width=462) # dataset 17 - 19
        elif self.dataset_list[index] in [20,21,22]:
          x = transforms.functional.crop(x,top=70,left=145,height=462,width=462) # dataset 20-21
        elif self.dataset_list[index] in [23]:
          x = transforms.functional.crop(x,top=30,left=250,height=462,width=462) # dataset 23  
        else:
            x = transforms.functional.crop(x,top=113,left=395,height=288,width=288)
        '''
        
        if self.include_torque:
            y = self.label_array[index][1:7] #include torque
        else:
            y = self.label_array[index][1:4] #remove the timestamp
        
        if self.transform:
            x = self.transform(x)

        if self.custom_state is None:
            x2 = self.label_array[index][7:]
        else:
            x2 = self.label_array[index][self.custom_state]

        return x, x2, y 
    
    def generate_image_list(self,image_dir,data_sets):
        '''Generates the file list of images, accounting for exluded sets'''
        
        # get the list of folders for the images
        image_folder_list = natsort.humansorted(glob.glob(image_dir+self.image_folder_prefix))
        file_list = []
        dataset_list = []
        
        if data_sets is not None:
            for i in data_sets:
                new_files = natsort.humansorted(glob.glob(image_folder_list[i-1]+"/*.jpg"))
                file_list = file_list + new_files
                dataset_list = dataset_list + [i for n in range(len(new_files))]
        else:
            for i in range(len(image_folder_list)):
                new_files = natsort.humansorted(glob.glob(image_folder_list[i]+"/*.jpg"))
                file_list = file_list + natsort.humansorted(glob.glob(image_folder_list[i]+"/*.jpg")) 
                dataset_list = dataset_list + [i+1 for n in range(len(new_files))]
        
        return file_list,dataset_list
            
    def read_labels(self,label_dir,data_sets):
        '''loads all the label data, accounting for excluded sets'''
        
        file_list = natsort.humansorted(glob.glob(label_dir + "/labels_full_*.txt"))
        label_list = []
        
        if data_sets is not None:
            for i in data_sets:
                data = np.loadtxt(file_list[i-1],delimiter=",")
                label_list.append(data)
        else:
            for i in range(len(file_list)):
                data = np.loadtxt(file_list[i],delimiter=",")
                label_list.append(data)
        
        labels = np.concatenate(label_list,axis=0)
        
        return labels

