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
    def __init__(self, label_dir, image_dir, data_sets = None, transform=None, crop_list = [], eval_params = None , include_torque = False, custom_state = None):
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
        
        if eval_params is None:
            self.mean,self.stdev = self.normalize_state()
        else:
            mean,std = eval_params
            self.mean,self.stdev = self.normalize_state(mean=mean,stdev=std)
        
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

    def normalize_state(self,mean=None,stdev=None):
        '''
        Finds the mean and standard deviation of the dataset and applies
        it all values.
        
        Returns the mean and standard deviation

        '''
        if mean is None and stdev is None:
            mean = np.mean(self.label_array,axis=0)
            stdev = np.std(self.label_array,axis=0)
            
        self.raw_label_array = self.label_array
        self.label_array = (self.label_array-mean)/stdev
        
        return mean,stdev

class StateDataset(data.Dataset):
    
    '''Characterizes a dataset for PyTorch'''
    def __init__(self, label_dir, data_sets = None, eval_params = None , include_torque = False, custom_state = None):
        '''
        Initialization
           exclude_index is a list denoting which datasets to exclude indexed
           from 1
        '''
        self.label_dir = label_dir
        self.include_torque = include_torque
        self.custom_state= custom_state
        self.label_array = self.read_labels(self.label_dir,data_sets)
        if eval_params is None:
            self.mean,self.stdev = self.normalize_state()
        else:
            mean,std = eval_params
            self.mean,self.stdev = self.normalize_state(mean=mean,stdev=std)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_array)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.custom_state is None:
            x = self.label_array[index][7:]
        else:
            x = self.label_array[index][self.custom_state]
        if self.include_torque:
            y = self.label_array[index][1:7] #include torque
        else:
            y = self.label_array[index][1:4] #remove the timestamp
        
        dummy = 0 # so that the train loop can still execute
        
        return dummy, x, y
   
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
    
    def normalize_state(self,mean=None,stdev=None):
        '''
        Finds the mean and standard deviation of the dataset and applies
        it all values.
        
        Returns the mean and standard deviation

        '''
        if mean is None and stdev is None:
            mean = np.mean(self.label_array,axis=0)
            stdev = np.std(self.label_array,axis=0)
            
        self.raw_label_array = self.label_array
        self.label_array = (self.label_array-mean)/stdev
        
        return mean,stdev
    
def init_dataset(train_list,val_list,test_list,model_type,config_dict):
    
    file_dir = config_dict['file_dir']
    include_torque = config_dict['include_torque']
    custom_state = config_dict['custom_state']
    batch_size = config_dict['batch_size']
    
    if model_type == "S":
        train_set = StateDataset(file_dir,
                                 data_sets=train_list,
                                 include_torque = include_torque,
                                 custom_state= custom_state)
        val_set = StateDataset(file_dir,
                                data_sets=val_list,
                                eval_params=(train_set.mean,train_set.stdev),
                                include_torque = include_torque,
                                custom_state= custom_state)
        test_set = StateDataset(file_dir,
                                data_sets=test_list,
                                eval_params=(train_set.mean,train_set.stdev),
                                include_torque = include_torque,
                                custom_state= custom_state)
    else:
        crop_list = config_dict['crop_list']
        trans_function = config_dict['trans_function']
        
        train_set = ImgDataset(file_dir,file_dir,
                               data_sets=train_list,
                               transform = trans_function,
                               crop_list=crop_list,
                               include_torque= include_torque,
                               custom_state= custom_state)
        val_set = ImgDataset(file_dir,file_dir,
                              data_sets=val_list,
                              transform = trans_function,
                              crop_list=crop_list,
                              eval_params=(train_set.mean,train_set.stdev),
                              include_torque= include_torque,
                              custom_state= custom_state)
        test_set = ImgDataset(file_dir,file_dir,
                              data_sets=test_list,
                              transform = trans_function,
                              crop_list=crop_list,
                              eval_params=(train_set.mean,train_set.stdev),
                              include_torque= include_torque,
                              custom_state= custom_state)
        
    train_loader = data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_loader = data.DataLoader(val_set,batch_size=batch_size,shuffle=False)
    test_loader = data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
    
    dataloaders_dict = {'train':train_loader,'val':val_loader,'test':test_loader}
    dataset_sizes = {'train':len(train_set),'val':len(val_set),'test':len(test_set)}
    
    return dataloaders_dict,dataset_sizes
    