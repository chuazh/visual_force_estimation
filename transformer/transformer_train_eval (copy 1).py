#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:38:30 2021

@author: charm

transformer training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models,transforms
from torch.utils import data
from torch.optim import lr_scheduler

import numpy as np
import glob
import natsort
import os 
import time
import copy
import pickle

# create sequential loader for state data

class sequential_dataset(data.Dataset):
    
    def __init__(self,filedir,lookback,lookforward,skips=0,predict_current=True,data_sets=None,eval_params=None):
        
        self.lb = lookback
        self.lf = lookforward
        self.skips = skips
        self.pred_c = predict_current
        self.label_array,self.lookup = self.read_labels(filedir,data_sets)
        
        if eval_params is None:
            self.mean,self.stdev = self.normalize_state()
        else:
            mean,std = eval_params
            self.mean,self.stdev = self.normalize_state(mean=mean,stdev=std)
        
    def __len__(self):
        
        return(len(self.lookup))
    
    def __getitem__(self,index):
        
        idx = self.lookup[index] #use the lookup table to find the true index to fetch from our label_array
        if self.skips+1>self.lb:
            print('warning: skip step is larger than look back range!')
        
        back_range=np.flip(np.arange(idx,idx-self.lb-1,step=1-self.skips).astype(int))
        forward_range = np.arange(idx,idx+self.lf+1,step=1).astype(int)
        
        input_data = (self.label_array[back_range])[:,7:61]
        timestamp = np.arange(input_data.shape[0]) # we create a timestamp because the transformer has no concept of time.
        input_data = np.hstack((timestamp.reshape((-1,1)),input_data))
        
        if self.pred_c:
            pred_data = (self.label_array[forward_range])[:,1:4] #try to predict current force plus into the future
        else:
            pred_data = (self.label_array[forward_range])[1:,1:4] #try to predict only the future force (useful in the current force is average from the previous future predictions?)
        
        force_history = (self.label_array[back_range])[:,1:4]
        
        return input_data,pred_data,force_history
    
    def read_labels(self,label_dir,data_sets):
        '''
        Loads all the label data, accounting for excluded sets.
        Generates a look up table to constrain get_item to the right range.
        '''
        
        file_list = natsort.humansorted(glob.glob(label_dir + "/labels_*.txt"))
        label_list = []
        start_index = 0
        lookup= np.empty((0,))
        
        if data_sets is not None:
            for i in data_sets:
                data = np.loadtxt(file_list[i-1],delimiter=",")
                true_index = np.arange(start_index+self.lb,start_index+data.shape[0]-self.lf)
                label_list.append(data)
                start_index += data.shape[0]
                lookup = np.hstack((lookup,true_index))
        else:
            for i in range(len(file_list)):
                data = np.loadtxt(file_list[i],delimiter=",")
                true_index = np.arange(start_index+self.lb,start_index+data.shape[0]-self.lf)
                label_list.append(data)
                start_index += data.shape[0]
                lookup = np.hstack((lookup,true_index))
        
        labels = np.concatenate(label_list,axis=0)
        
        return labels,lookup
    
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
    
    def print_lookup(self):
        dataset_index = np.arange(len(self.lookup))
        print(dataset_index)
        print(self.lookup)
        
        return dataset_index,self.lookup

# TRANSFORMER MODEL SPEC
    
class TransformerModel(nn.Module):

  def __init__(self,lookahead=10):

    super(TransformerModel,self).__init__()

    input_dims = 55
    enc_input_dims = 1024
    self.enc_input_dims = enc_input_dims
    num_heads = 8
    enc_ff_dims = 2048
    num_enc_layers = 12
    dropout = 0.1
    self.lookahead = lookahead

    self.encoder = nn.Linear(input_dims,enc_input_dims)
    encoder_layers = nn.TransformerEncoderLayer(enc_input_dims,num_heads,enc_ff_dims,dropout)
    self.tranformer_encoder = nn.TransformerEncoder(encoder_layers,num_enc_layers)
    
    decoder_layers = nn.TransformerDecoderLayer(enc_input_dims,num_heads,enc_ff_dims,dropout)
    self.transformer_decoder = nn.TransformerDecoder(decoder_layers,num_enc_layers)
    
    self.fc1 = nn.Linear(enc_input_dims,3)

  def forward(self,input,device):
    x = self.encoder(input) * np.sqrt(self.enc_input_dims)
    x = self.tranformer_encoder(x)
    x0 = torch.zeros(self.lookahead+1,x.size()[1],x.size()[2]).to(device,dtype=torch.float)
    x = self.transformer_decoder(x0,x)
    out = self.fc1(nn.functional.relu(x))

    return out

# TRAINING LOOP FUNCTION

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_loss = np.Inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            it = 1
            batch_size = 0

            for inputs,labels,force_history in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()
                inputs = torch.transpose(inputs,0,1) # just transpose the batch to the second dimension
                #labels = torch.transpose(labels,0,1) # just transpose the batch to the second dimension
                inputs = inputs.to(device,dtype=torch.float)
                labels = labels.to(device,dtype=torch.float)

                # forward
                # track history if only in train
                if phase == 'train':
                  torch.set_grad_enabled(True)
                  outputs= model(inputs,device)
                  loss = criterion(torch.transpose(outputs,0,1),labels)
                  loss.backward()
                  #xm.optimizer_step(optimizer,barrier=True)
                  optimizer.step()
                else :
                  torch.set_grad_enabled(False)
                  outputs=model(inputs,device)
                  loss = criterion(torch.transpose(outputs,0,1),labels)
                
                # statistics
                running_loss += loss.item() #* inputs.size(0) # multiply by the number of elements to get back the total loss, usually the loss function outputs the mean
                batch_size += inputs.size(1)
                avg_loss = running_loss/batch_size
                if it%10 == 0:
                  print('average loss for iteration ' + str(it)+ ' : ' + str(avg_loss))
                it += 1

            epoch_loss = running_loss / dataset_sizes[phase] #divide by the total size of our dataset to get the mean loss per instance
            
            if phase== 'train':
                train_losses[epoch] = epoch_loss
                scheduler.step(epoch_loss)
            else:
                val_losses[epoch] = epoch_loss
                
            
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print('Saving model... current loss:' + str(round(epoch_loss,5)) + ' < best loss: ' + str(round(best_loss,5)))
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(model.state_dict())
                #for k, v in best_model_wts.items():
                  #best_model_wts[k] = v.cpu()
                print("Backing up the model")
                temp_file = open("best_modelweights_S_transformer.dat","wb")
                torch.save(model.state_dict(),temp_file)
                #xm.save(model.state_dict(),temp_file)
                #!cp best_modelweights_transformer.dat -d '/content/drive/My Drive/Zonghe Chua Research Folder/Data Driven Surgical Study/dvrk data 081320'
                #xm.save(model.state_dict(),temp_file)
                
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    temp_file.close()
    temp_file = open("best_modelweights_S_transformer.dat","rb")
    model.load_state_dict(torch.load(temp_file))
    
    return model, train_losses, val_losses

def compute_loss_metrics(predictions,labels,max_f,min_f,condition,model_type):
    print()
    print("Summary Performance Statistics")
    print("-"*10)
    # compute MSEloss
    mean_error = np.mean(np.linalg.norm(predictions-labels,axis=1))
    print("Mean Resultant Force Error: {0:.3f}".format(mean_error))
    
    # compute normalized mean loss #in accurate.
    range_divisor = np.max(np.linalg.norm(labels,axis=1))-np.min(np.linalg.norm(labels,axis=1))
    normed_mean_error = np.mean(np.linalg.norm(predictions-labels,axis=1)/range_divisor,axis=0)
    print("Normalized Mean Resultant Force Error: {0:.3f}".format(normed_mean_error))
    
    # per axis errors
    per_axis_RMSE = np.sqrt(np.mean((predictions-labels)**2,axis=0))
    print("Per Axis RMSE: x:{0:.3f}, y:{1:.3f},z:{2:.3f}".format(per_axis_RMSE[0],per_axis_RMSE[1],per_axis_RMSE[2]))
    
    #normalize error
    range_divisor = max_f-min_f
    normed_axis_RMSE = np.sqrt(np.mean((predictions-labels)**2,axis=0))/range_divisor
    print("Per Axis normalized RMSE: x:{0:.3f}, y:{1:.3f},z:{2:.3f}".format(normed_axis_RMSE[0],normed_axis_RMSE[1],normed_axis_RMSE[2]))
    print(' ')
    
    output_dict = {'model': model_type,
                   'condition': condition, 
                   'ME': mean_error,
                   'nME': normed_mean_error,
                   'Per Axis RMSEx': per_axis_RMSE[0],
                   'Per Axis RMSEy': per_axis_RMSE[1],
                   'Per Axis RMSEz': per_axis_RMSE[2],
                   'Per Axis nRMSEx':normed_axis_RMSE[0],
                   'Per Axis nRMSEy':normed_axis_RMSE[1],
                   'Per Axis nRMSEz':normed_axis_RMSE[2]}
    
    #return pd.DataFrame.from_dict(output_dict)
    return output_dict




#%%

file_location = "../../experiment_data"

train_set = sequential_dataset(file_location,lookback=60,lookforward=0,skips=14,data_sets= [1,3,5,7,8,10,12,14,15,17,19,21,41,42])
mn = train_set.mean
sd = train_set.stdev
val_set = sequential_dataset(file_location,lookback=60,lookforward=0,skips=14,data_sets=[2,6,9,13,16,20,44],eval_params=(mn,sd))

dataloaders = {}
dataloaders['train'] = data.DataLoader(train_set,batch_size=64,shuffle=True)
dataloaders['val'] = data.DataLoader(val_set,batch_size=64,shuffle=False)
data_size = {}
data_size['train'] = len(train_set)
data_size['val'] = len(val_set)
#%%
model = TransformerModel(lookahead=0)
criterion = nn.MSELoss(reduction="sum")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device,dtype=torch.float)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
model_data = train_model(model,criterion,optimizer,dataloaders,data_size,num_epochs=30)

#%% LOAD MODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    model = model_data[0]
except:
    model = TransformerModel(lookahead=0)
    model.load_state_dict(torch.load('best_modelweights_S_transformer.dat'))
model = model.to(device)
model.eval()

#%% TEST PLOT and INFERENCE

def inference(loader,model,device,correction=None):
    force_preds = np.empty((0,3))
    gt = np.empty((0,3))
    if correction is not None: # takes a tuple of correction values
            mn,sd = correction
    with torch.no_grad():
        for inputs,labels,force_history in test_loader:
            inputs = torch.transpose(inputs,0,1) # just transpose the batch to the second dimension
            inputs = inputs.to(device,dtype=torch.float)
            pred = model(inputs,device)
            pred = pred.cpu().squeeze().numpy()
            labels = labels.squeeze().numpy()
            if correction is not None:
                pred = (pred * sd[1:4]) + mn[1:4]
                labels = (labels * sd[1:4]) + mn[1:4]
            
            force_preds = np.vstack((force_preds,pred))
            gt = np.vstack((gt,labels))
    
    return force_preds, gt

test_set = sequential_dataset(file_location,lookback=60,lookforward=0,skips=14,data_sets=[45],eval_params=(mn,sd))
test_loader = data.DataLoader(test_set,batch_size=64,shuffle = False)

model_preds, model_labels = inference(test_loader,model,device)

import matplotlib.pyplot as plt

fig,ax = plt.subplots(3,1,sharex=True)
for i in range(3):
    ax[i].plot(model_labels[:,i])
    ax[i].plot(model_preds[:,i])

#%%
    
test_list_full =  [4,11,18,
                   22,23,
                   24,25,
                   26,27,
                   28,29,
                   32,33,
                   34,36,
                   37,38,39,
                   45,46,47]


condition_list = ['center','right','left',
                  'right_less','right_less',
                  'right_more','right_more',
                  'left_less','left_less',
                  'left_more','left_more',
                  'new_tool','new_tool',
                  'new_material','new_material',
                  'center','right','left',
                  'z_mid','z_high','z_low']
model_type = "Trans."
max_force = []
min_force = []
for test in test_list_full:
    forces = np.loadtxt(file_location+'/labels_'+str(test)+'.txt',delimiter = ",")[:,1:4]
    test_max = np.max(forces,axis=0)
    test_min = np.min(forces,axis=0)
    max_force.append(test_max)
    min_force.append(test_min)

metrics_list = []
preds_list = []
gt_list = []
for i,(test,condition) in enumerate(zip(test_list_full,condition_list)):
    test_set = sequential_dataset(file_location,lookback=60,lookforward=0,skips=14,data_sets=[test],eval_params=(mn,sd))
    test_loader = data.DataLoader(test_set,batch_size=64,shuffle = False)
    model_preds,model_labels = inference(test_loader,model,device,correction = (mn,sd))
    metrics = compute_loss_metrics(model_preds,model_labels,max_force[i],min_force[i],condition,model_type)
    metrics_list.append(metrics)
    preds_list.append(model_preds)
    gt_list.append(model_labels)
    
#%% Create Dataframes
import pandas as pd    
df_metrics = pd.DataFrame(metrics_list)    
test_numbering_list = [1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,2,2,2,1,1,1]

df_metrics['test_number'] = test_numbering_list
df_metrics['model']='S_T'

df_summary = df_metrics.groupby(['condition']).mean()
df_merge = pd.melt(df_metrics,id_vars=['condition','model'],value_vars=['Per Axis RMSEx','Per Axis RMSEy','Per Axis RMSEz'],var_name = 'metric',value_name='value')
