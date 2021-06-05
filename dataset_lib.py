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
from scipy.spatial.transform import Rotation as R
import scipy.signal as sig
import copy

class RNN_ImgDataset(data.Dataset):
    
    def __init__(self,filedir,lookback,trans_function,crop_list=[],skips=0,data_sets=None):
        
        self.lookback = lookback
        self.skips = skips
        self.image_folder_prefix = '/imageset'
        self.file_dir = filedir
        self.image_dir = filedir
        self.label_array,self.lookup = self.read_labels(filedir,data_sets)
        self.image_list,self.dataset_list = self.generate_image_list(self.image_dir,data_sets)
        self.include_torque= False
        self.grayscale_convert = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale()])
        self.trans_function = trans_function
        
        self.crop_list = crop_list
    
    def __len__(self):
        return(len(self.lookup))
    
    def __getitem__(self,index):
        
        idx = int(self.lookup[index])
        
        if self.skips+1>self.lookback:
            print("Warning: Skip step is larger than look back range.")
        
        # generate spacetime images
        # grab the dataset in which this sequence belongs to
        image_mean_idx = self.dataset_list[index]
        # get the image mean
        img_mean = transforms.ToTensor()(im.open(self.file_dir+self.image_folder_prefix+"_"+str(image_mean_idx)+"/image_mean.jpg"))
        # the back range give a list of indexes we should load
        back_range = np.flip(np.arange(idx,idx-self.lookback-1,step=-1-self.skips).astype(int)) 
        img_list = []
        for img_idx in back_range: 
            x = transforms.ToTensor()(im.open(self.image_list[img_idx]))-img_mean + 127 # load images and subtract the mean
            
            x = self.grayscale_convert(x) # convert to gray scale
            if len(self.crop_list)> 0 : # perform the image crop
                #use the crop list
                t = self.crop_list[image_mean_idx][0]
                l = self.crop_list[image_mean_idx][1]
                h = self.crop_list[image_mean_idx][2]
                w = self.crop_list[image_mean_idx][3]
                x = transforms.functional.crop(x,top=t,left=l,height=h,width=w)
            else:
                #Use the default crop
                #x = transforms.functional.crop(x,top=57,left=320,height=462,width=462)
                x = transforms.functional.crop(x,top=100,left=320,height=250,width=250) #new dataset
            
            x = transforms.Resize((224,224))(x)
            x = transforms.ToTensor()(x).squeeze()
            img_list.append(x)
        
        x = torch.stack(img_list,axis=0)
        # end generate spacetime images
        
        '''
        x = transforms.ToTensor()(im.open(self.image_list[index]))
        x = self.trans_function(x) # convert to tensor and normalize by image net
        '''
        y = self.label_array[idx][1:4]
        x2 = 0
        
        return x,x2,y
        
    def read_labels(self,label_dir,data_sets):
        
        file_list = natsort.humansorted(glob.glob(label_dir + "/labels_*.txt"))
        label_list = []
        start_index = 0
        lookup = np.empty((0,))
        
        if data_sets is not None:
            for i in data_sets:
                data = np.loadtxt(file_list[i-1],delimiter=",")
                true_index = np.arange(start_index+self.lookback, start_index+data.shape[0])
                label_list.append(data)
                start_index += data.shape[0]
                lookup = np.hstack((lookup,true_index))
        else:
            for i in range(len(file_list)):
                data=np.loadtxt(file_list[i],delimiter=",")
                true_index = np.arange(start_index+self.lookback,start_index+data.shape[0])
                label_list.append(data)
                start_index += data.shape[0]
                lookup = np.hstack((lookup,true_index))
        
        labels=np.concatenate(label_list,axis=0)
        
        return labels,lookup
    
    def generate_image_list(self,image_dir,data_sets):
        '''Generates the file list of images, accounting for exluded sets'''
        
        # get the list of folders for the images
        image_folder_list = natsort.humansorted(glob.glob(image_dir+self.image_folder_prefix+"*"))
        file_list = []
        dataset_list = []
        
        if data_sets is not None:
            for i in data_sets:
                new_files = natsort.humansorted(glob.glob(image_folder_list[i-1]+"/img*.jpg"))
                file_list = file_list + new_files
                dataset_list = dataset_list + [i for n in range(len(new_files))]
        else:
            for i in range(len(image_folder_list)):
                new_files = natsort.humansorted(glob.glob(image_folder_list[i]+"/img*.jpg"))
                file_list = file_list + natsort.humansorted(glob.glob(image_folder_list[i]+"/img*.jpg")) 
                dataset_list = dataset_list + [i+1 for n in range(len(new_files))]
        
        return file_list,dataset_list    



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
    def __init__(self, label_dir, image_dir, data_sets = None, transform=None, crop_list = [], eval_params = None , include_torque = False, custom_state = None, crop_jitter = False,color_jitter=False,rand_erase=False):
        '''
        Initialization
           exclude_index is a list denoting which datasets to exclude indexed
           from 1
        '''
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.transform = transform
        self.crop_list = crop_list
        self.crop_jitter = crop_jitter
        self.color_jitter = color_jitter
        self.rand_erase = rand_erase
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
        
        self.func_colorjit = transforms.ColorJitter(brightness = (0.5,1.5), contrast =(0.5,1.5))
        self.func_randerase = transforms.RandomErasing(p=0.25,scale=(0.01,0.05),ratio=(0.3,3.3))
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_array)

    def __getitem__(self, index):
        'Generates one sample of data'
        x = im.open(self.image_list[index])
        
        if self.color_jitter:
            x = self.func_colorjit(x)
            
        jitter = [0,0]
        if self.crop_jitter:
            for i in range(2):
                jitter[i] = np.random.randint(-10,10)
        
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

        if self.rand_erase:
            x = self.func_randerase(x)

        if self.custom_state is None:
            x2 = self.label_array[index][7:61]
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
                new_files = natsort.humansorted(glob.glob(image_folder_list[i-1]+"/img*.jpg"))
                file_list = file_list + new_files
                dataset_list = dataset_list + [i for n in range(len(new_files))]
        else:
            for i in range(len(image_folder_list)):
                new_files = natsort.humansorted(glob.glob(image_folder_list[i]+"/img*.jpg"))
                file_list = file_list + natsort.humansorted(glob.glob(image_folder_list[i]+"/img*.jpg")) 
                dataset_list = dataset_list + [i+1 for n in range(len(new_files))]
        
        return file_list,dataset_list
            
    def read_labels(self,label_dir,data_sets):
        '''loads all the label data, accounting for excluded sets'''
        
        file_list = natsort.humansorted(glob.glob(label_dir + "/labels_*.txt"))
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

    def mask_labels(self,feature_set):
        
        '''masks the state dataset by setting the features to zero'''
        self.label_array[:,feature_set] = 0.0

class StateDataset(data.Dataset):
    
    '''Characterizes a dataset for PyTorch'''
    def __init__(self, label_dir, data_sets = None, eval_params = None , include_torque = False, spatial_force=False , custom_state = None):
        '''
        Initialization
           exclude_index is a list denoting which datasets to exclude indexed
           from 1
        '''
        self.label_dir = label_dir
        self.include_torque = include_torque
        self.custom_state= custom_state
        self.spatial_force = spatial_force
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
            x = self.label_array[index][7:61]
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
        
        file_list = natsort.humansorted(glob.glob(label_dir + "/labels_*.txt"))
        label_list = []
        
        if data_sets is not None:
            for i in data_sets:
                print(file_list[i-1])
                data = np.loadtxt(file_list[i-1],delimiter=",")
                #b,a = sig.butter(3,5/15)
                #data[:,1:4] = sig.filtfilt(b,a,copy.copy(data[:,1:4]),axis=0)
                if self.spatial_force:
                    data = realign_forces(data, np.array([10,11,12,13]), np.array([55,56,57]))
                label_list.append(data)
        else:
            for i in range(len(file_list)):
                data = np.loadtxt(file_list[i],delimiter=",")
                if self.spatial_force:
                    data = realign_forces(data, np.array([10,11,12,13]), np.array([55,56,57]))
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
    
    def mask_labels(self,feature_set):
        
        '''masks the state dataset by setting the features to zero'''
        self.label_array[:,feature_set] = 0.0
      
    
def realign_forces(dataset,pose_idx,psm_force_idx):
    
    '''Function align the wrench forces from the ee frame to the spatial/base frame
    Input: dataset = N x F numpy array where F is the full number of features
           pose_idx= column index containing the pose data as quaternions
           force_idx = column index containing the body wrench data
    Output: dataset NxF numpy array where the force_idx columns are overwritten with the transformed forces. 
    '''
    
    # retrieve the relevant data from the dataset
    pose_quaternions = R.from_quat(dataset[:,pose_idx]) # this is the ee pose in spatial frame
    body_forces = dataset[:,psm_force_idx] # this are cartesian forces in ee frame
    
    # we need to rotate the body forces into the spatial frame
    pose_quaternions_inv = pose_quaternions.inv() # get the reverse mappings : body to spatial
    # apply the rotations to the forces
    spatial_forces = pose_quaternions_inv.apply(body_forces)
    
    # overwrite the forces in the dataset
    dataset[:,psm_force_idx] = spatial_forces
    
    return dataset
    
    
def init_dataset(train_list,val_list,test_list,model_type,config_dict,augment=False):
    
    file_dir = config_dict['file_dir']
    include_torque = config_dict['include_torque']
    custom_state = config_dict['custom_state']
    batch_size = config_dict['batch_size']
    spatial_force = config_dict['spatial_forces']
    
    if model_type == "S":
        train_set = StateDataset(file_dir,
                                 data_sets=train_list,
                                 include_torque = include_torque,
                                 custom_state= custom_state,
                                 spatial_force=spatial_force)
        val_set = StateDataset(file_dir,
                                data_sets=val_list,
                                eval_params=(train_set.mean,train_set.stdev),
                                include_torque = include_torque,
                                custom_state= custom_state,
                                spatial_force=spatial_force)
        test_set = StateDataset(file_dir,
                                data_sets=test_list,
                                eval_params=(train_set.mean,train_set.stdev),
                                include_torque = include_torque,
                                custom_state= custom_state,
                                spatial_force=spatial_force)
    elif model_type == "V_RNN":
        crop_list = config_dict['crop_list']
        trans_function = config_dict['trans_function']
        lookback = 20
        train_set = RNN_ImgDataset(file_dir,
                                   lookback,
                                   trans_function,
                                   crop_list=[],
                                   skips=9,
                                   data_sets=train_list)
        val_set = RNN_ImgDataset(file_dir,
                                   lookback,
                                   trans_function,
                                   crop_list=[],
                                   skips=9,
                                   data_sets=val_list)
        test_set = RNN_ImgDataset(file_dir,
                                   lookback,
                                   trans_function,
                                   crop_list=[],
                                   skips=9,
                                   data_sets=test_list)
    else:
        crop_list = config_dict['crop_list']
        trans_function = config_dict['trans_function']
        
        train_set = ImgDataset(file_dir,file_dir,
                               data_sets=train_list,
                               transform = trans_function,
                               crop_list=crop_list,
                               include_torque= include_torque,
                               custom_state= custom_state,
                               color_jitter=augment,
                               crop_jitter=augment,
                               rand_erase=augment)
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
    