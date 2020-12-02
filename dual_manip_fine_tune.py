#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:02:53 2020

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
import glob
import natsort

#%% Create DataLoader for First PSM

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

#%% Create DataLoader for the Second PSM

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

#%% define custom dual loader
from torch.utils import data

class StateDataset(data.Dataset):
    
    '''Characterizes a dataset for PyTorch'''
    def __init__(self, label_dir1, label_dir2, eval_params1, eval_params2, data_sets = None):
        '''
        Initialization
           exclude_index is a list denoting which datasets to exclude indexed
           from 1
        '''
        self.label_dir1 = label_dir1
        self.label_dir2 = label_dir2
        
        self.raw_label_array1 = self.read_labels(self.label_dir1,data_sets)
        self.raw_label_array2 = self.read_labels(self.label_dir2,data_sets)
        
        self.mean1,self.stdev1 = eval_params1
        self.mean2,self.stdev2 = eval_params2
        
        self.label_array1 = self.normalize_state(self.raw_label_array1,mean=mean1,stdev=std1)
        self.label_array2 = self.normalize_state(self.raw_label_array2,mean=mean2,stdev=std2)
        
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_array1)

    def __getitem__(self, index):
        'Generates one sample of data'

        x1 = self.label_array1[index][7:61]
        x2 = self.label_array2[index][7:61]
        
        y = self.raw_label_array1[index][1:4] #remove the timestamp
        
        
        return x1, x2, y
   
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
    
    def normalize_state(self,array,mean=None,stdev=None):
        '''
        Finds the mean and standard deviation of the dataset and applies
        it all values.
        
        Returns the mean and standard deviation

        '''
        if mean is None and stdev is None:
            mean = np.mean(self.label_array,axis=0)
            stdev = np.std(self.label_array,axis=0)
            
        label_array = (array-mean)/stdev
        
        return label_array

#%%
        
dataset = {}
label_dir1 = "../dual_manip_data/PSM1"
label_dir2 = "../dual_manip_data/PSM2"
dataset['train'] = StateDataset(label_dir1, label_dir2, (mean1,std1), (mean2,std2), data_sets = [1,4])
dataset['val'] = StateDataset(label_dir1, label_dir2, (mean1,std1), (mean2,std2), data_sets = [3])
dataset['test'] = StateDataset(label_dir1, label_dir2, (mean1,std1), (mean2,std2), data_sets = [2])

dataloaders = {}
dataloaders['train'] = data.DataLoader(dataset['train'],batch_size=32,shuffle=True)
dataloaders['val'] = data.DataLoader(dataset['val'],batch_size=32,shuffle=True)
dataloaders['test'] = data.DataLoader(dataset['test'],batch_size=32,shuffle=True)

loader_sizes = {}
loader_sizes['train'] = len(dataset['train'])
loader_sizes['val'] = len(dataset['val'])

#%% define custom model
    
class dual_model(nn.Module):
  
  def __init__(self,model1,model2,miu1,sig1,miu2,sig2):
    super(dual_model,self).__init__()
    self.m1 = model1
    self.m2 = model2
    self.miu1 = torch.from_numpy(miu1)
    self.sig1 = torch.from_numpy(sig1)
    self.miu2 = torch.from_numpy(miu2)
    self.sig2 = torch.from_numpy(sig2)

  def transfer(self,device):
    #self.m1.to(device,torch.float)
    #self.m2.to(device,torch.float)
    self.miu1 = self.miu1.to(device,torch.float)
    self.sig1 = self.sig1.to(device,torch.float)
    self.miu2 = self.miu2.to(device,torch.float)
    self.sig2 = self.sig2.to(device,torch.float)

  def forward(self,input1,input2):
    out1 = self.m1(input1)
    out2 = self.m2(input2)
    
    #unnormalize the predictions
    out1 = (out1*self.sig1)+self.miu1
    out2 = (out2*self.sig2)+self.miu2
    x = out1+out2
    
    return x

model1  = mdl.StateModel(54, 3)
model1.load_state_dict(torch.load("best_modelweights_S_PSM1.dat"))
model2  = mdl.StateModel(54, 3)
model2.load_state_dict(torch.load("best_modelweights_S.dat"))

full_model = dual_model(model1,model2,mean1[1:4],std1[1:4],mean2[1:4],std2[1:4])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
full_model.transfer(device)
#%%
import torch.optim as opt
weight_file1 = "best_modelweights_S_PSM1_ft.dat"
weight_file2 = "best_modelweights_S_PSM2_ft.dat"

criterion = nn.MSELoss(reduction='sum')
# define optimization method
optimizer = opt.Adam(full_model.parameters(),lr=0.001,weight_decay=0)
full_model,train_history,val_history,_ = train_model(full_model,
                                                         criterion, optimizer,
                                                         dataloaders,
                                                         loader_sizes,  
                                                         num_epochs=100,
                                                         L1_loss=0.001,
                                                         model_type= model_type,
                                                         weight_file1=weight_file1,
                                                         weight_file2=weight_file2,
                                                         suppress_log=False)


#%% CUSTOM Training Loop
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except:
    pass

import time
from torch.utils.tensorboard import SummaryWriter

def train_model(model, criterion, optimizer, dataloader1, dataset_sizes, num_epochs=10, model_type = "VS", weight_file1 = "best_modelweights.dat", weight_file2 = "best_modelweights.dat", L1_loss = 0 ,suppress_log=False, hyperparam_search = False, use_tpu=False, tensorboard = True):
    
    if use_tpu:
        print("using TPU acceleration, model and optimizer should already be loaded onto tpu device")
        device = xm.xla_device()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("using GPU acceleration")
    
        model = model.to(device,dtype=torch.float)
        

    since = time.time()
    best_loss = np.Inf
    
    #train_losses = np.zeros(num_epochs*dataset_sizes['train'])
    #val_losses = np.zeros(num_epochs*dataset_sizes['val'])
    train_losses = np.zeros(num_epochs*len(dataloader1['train']))
    val_losses = np.zeros(num_epochs*len(dataloader1['val']))
    
    it_val = 0
    it_train = 0
    
    if tensorboard:
        writer = SummaryWriter()

    for epoch in range(num_epochs):
        if suppress_log==False:
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                # initialize the predictions
                predictions = np.empty((0,3))

            running_loss = 0.0

            # Iterate over data.
            batch_size = 0
            it = 1
            
            for in1 in dataloader1[phase]:
                # zero the parameter gradients
                
                inputs,aug_inputs,labels = in1
                
                optimizer.zero_grad()
                
                if model_type !="S":
                    inputs = inputs.to(device,dtype=torch.float)
                
                if (model_type != "V") or (model_type!="V_RNN"):
                    inputs = inputs.to(device,dtype=torch.float)
                    aug_inputs = aug_inputs.to(device,dtype=torch.float)
                    
                    
                labels = labels.to(device,dtype=torch.float)

                # forward
                # track history if only in train
                if phase == 'train':
                  torch.set_grad_enabled(True)
                  
                  if (model_type == "V") or (model_type=="V_RNN"):
                      outputs = model(inputs)
                  elif model_type == "VS":
                      outputs= model(inputs,aug_inputs)
                  else:
                      outputs = model(inputs,aug_inputs)
                  
                  loss = criterion(outputs,labels)
                  
                  if L1_loss:
                      L1 = 0
                      for param in model.parameters():
                          if param.requires_grad:
                              L1 += L1_loss*torch.sum(torch.abs(param))
                      loss = loss+L1 
                  
                  
                  loss.backward()
                  if use_tpu:
                      xm.optimizer_step(optimizer,barrier=True)
                  else:
                      optimizer.step()
                else :
                  torch.set_grad_enabled(False)
                  
                  if (model_type == "V") or (model_type=="V_RNN"):
                      outputs = model(inputs)
                  elif model_type == "VS":
                      outputs= model(inputs,aug_inputs)
                  else:
                      outputs = model(inputs,aug_inputs)
                      
                  loss = criterion(outputs,labels)
                  predictions = np.vstack((predictions,outputs.cpu().detach().numpy()))
                
                # statistics
                running_loss += loss.item() #* inputs.size(0) # multiply by the number of elements to get back the total loss, usually the loss function outputs the mean
                batch_size += inputs.size(0)
                avg_loss = running_loss/batch_size
                
                if phase== 'train':
                    train_losses[it_train] = avg_loss
                    if tensorboard:
                        writer.add_scalar('Loss/train',avg_loss,it_train)
                    it_train += 1
                else:
                    val_losses[it_val] = avg_loss
                    if tensorboard:
                        writer.add_scalar('Loss/val',avg_loss,it_val)
                    it_val += 1
                
                if it%100 == 0 and suppress_log==False:
                  print('average loss for batch ' + str(it)+ ' : ' + str(avg_loss))
            
                it +=1

            epoch_loss = running_loss / dataset_sizes[phase] #divide by the total size of our dataset to get the mean loss per instance
            
            if tensorboard:
                if phase=="train":
                    writer.add_scalar('ELoss/train',epoch_loss,epoch)
                if phase=="val":
                     writer.add_scalar('ELoss/val',epoch_loss,epoch)
                 
            
            if suppress_log==False:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                
                if hyperparam_search==False:
                    print('Saving model... current loss:' + str(round(epoch_loss,5)) + ' < best loss: ' + str(round(best_loss,5)))
                    print("Backing up the model")
                    temp_file = open(weight_file1,"wb")
                    torch.save(model.m1.state_dict(),temp_file)
                    temp_file = open(weight_file2,"wb")
                    torch.save(model.m2.state_dict(),temp_file)
                
                else:
                    print('current loss:' + str(round(epoch_loss,5)) + ' < best loss: ' + str(round(best_loss,5)))
                    
                best_loss = epoch_loss

        if suppress_log==False:
            time_elapsed = time.time() - since
            print('Epoch runtime {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    if hyperparam_search==False:
        temp_file.close()
        temp_file = open(weight_file1,"rb")
        model.m1.load_state_dict(torch.load(temp_file))
        temp_file = open(weight_file2,"rb")
        model.m2.load_state_dict(torch.load(temp_file))
    
    return model, train_losses, val_losses, best_loss     

