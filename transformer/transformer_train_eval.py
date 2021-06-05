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

os.environ["CUBLAS_WORKSPACE_CONFIG"]= ":16:8"

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
        
        back_range=np.flip(np.arange(idx,idx-self.lb-1,step=-1-self.skips).astype(int))
        forward_range = np.arange(idx,idx+self.lf+1,step=1).astype(int)
        
        input_data = (self.label_array[back_range])[:,7:61]
        timestamp = np.arange(input_data.shape[0]) # we create a timestamp because the transformer has no concept of time.
        #timestamp = (self.raw_label_array[back_range])[:,0]
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

  def forward(self,input_feature,device):
    x = self.encoder(input_feature) * np.sqrt(self.enc_input_dims)
    x = self.tranformer_encoder(x)
    x0 = torch.zeros(self.lookahead+1,x.size()[1],x.size()[2]).to(device,dtype=torch.float)
    x = self.transformer_decoder(x0,x)
    out = self.fc1(nn.functional.relu(x))

    return out

# LSTM

class LSTMModel(nn.Module):
    
    def __init__(self):
        
        super(LSTMModel,self).__init__()
        
        input_dims = 55
        hidden_size = 256
        layers = 4
        drop = 0.5
        self.lstm = nn.LSTM(input_dims, hidden_size,num_layers=layers,dropout=drop)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,3)
        
        
    def forward(self,input_feature,states):
        
        out,_ = self.lstm(input_feature,None) # ignore the state output
        out = out[-1,:,:]
        out = F.relu(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out).unsqueeze(1)
        
        return out

def pad_tensor(input_tensor,target_batch_size,pad_dim):
    
    tup = input_tensor.shape
    if tup[pad_dim] != target_batch_size:
        pad_len = target_batch_size-tup[pad_dim]
        if pad_dim == 0:
            padding = torch.zeros(pad_len,tup[1],tup[2])
        elif pad_dim ==1:
            padding = torch.zeros(tup[0],pad_len,tup[2])
        output_tensor = torch.cat((input_tensor,padding.double()),dim=pad_dim)
    else:
        output_tensor = input_tensor
    return output_tensor
    

# TRAINING LOOP FUNCTION

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, model_type,num_epochs=10):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_loss = np.Inf
    if model_type == "transformer":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    '''
    if model_type == "LSTM":
        btch = dataloaders['train'].batch_size
        states = (torch.zeros(model.lstm.num_layers,btch,model.lstm.hidden_size).to(device,torch.float),torch.zeros(model.lstm.num_layers,btch,model.lstm.hidden_size).to(device,torch.float))
    '''
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
                if model_type == "LSTM":
                    inputs = pad_tensor(inputs,dataloaders[phase].batch_size,1)
                    labels = pad_tensor(labels,dataloaders[phase].batch_size,0)
                #labels = torch.transpose(labels,0,1) # just transpose the batch to the second dimension
                inputs = inputs.to(device,dtype=torch.float)
                labels = labels.to(device,dtype=torch.float)
                
                # forward
                # track history if only in train
                if phase == 'train':
                  torch.set_grad_enabled(True)
                  if model_type == "transformer":
                      outputs= model(inputs,device)
                      outputs = torch.transpose(outputs,0,1)
                  else:
                      outputs = model(inputs,None)
                  loss = criterion(outputs,labels)
                  loss.backward()
                  #xm.optimizer_step(optimizer,barrier=True)
                  optimizer.step()
                else :
                  torch.set_grad_enabled(False)
                  
                  if model_type == "transformer":
                      outputs = model(inputs,device)
                      outputs = torch.transpose(outputs,0,1)
                  else:
                      outputs = model(inputs,None)
                      
                  loss = criterion(outputs,labels)
                
                # statistics
                running_loss += loss.item() #* inputs.size(0) # multiply by the number of elements to get back the total loss, usually the loss function outputs the mean
                batch_size += inputs.size(1)
                avg_loss = running_loss/batch_size
                if it%100 == 0:
                  print('average loss for iteration ' + str(it)+ ' : ' + str(avg_loss))
                it += 1

            epoch_loss = running_loss / dataset_sizes[phase] #divide by the total size of our dataset to get the mean loss per instance
            
            if phase== 'train':
                train_losses[epoch] = epoch_loss
                if model_type=="transformer":
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
                temp_file = open("best_modelweights_S_"+model_type+".dat","wb")
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
    temp_file = open("best_modelweights_S_"+model_type+".dat","rb")
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
    
'''note about the look back
our data is sampled at 30hz so any back calc has to be dones at 30hz?
a look back of 60 here would imply 60 samples back with a spacing interval of 0.033s per sample when performing inference.
'''
file_location = "../../experiment_data"
model_type = "LSTM"
train_set = sequential_dataset(file_location,lookback=60,lookforward=0,skips=0,data_sets= [1,3,5,7,8,10,12,14,15,17,19,21,41,42])
mn = train_set.mean
sd = train_set.stdev
val_set = sequential_dataset(file_location,lookback=60,lookforward=0,skips=0,data_sets=[2,6,9,13,16,20,44],eval_params=(mn,sd))

dataloaders = {}
dataloaders['train'] = data.DataLoader(train_set,batch_size=64,shuffle=True)
dataloaders['val'] = data.DataLoader(val_set,batch_size=64,shuffle=False)
data_size = {}
data_size['train'] = len(train_set)
data_size['val'] = len(val_set)
#%%
#model = TransformerModel(lookahead=0)
model = LSTMModel()
criterion = nn.MSELoss(reduction="sum")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device,dtype=torch.float)

lr = 1e-3#1e-5
optimizer = torch.optim.Adam(model.parameters(),lr=lr) #torch.optim.SGD(model.parameters(),lr=lr)
model_data = train_model(model,criterion,optimizer,dataloaders,data_size,model_type,num_epochs=30)

#%% LOAD MODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    model = model_data[0]
except:
    if model_type=="transformer":
        model = TransformerModel(lookahead=0)
    else:
        model = LSTMModel()
    model.load_state_dict(torch.load('best_modelweights_S_'+model_type+'.dat'))
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

test_set = sequential_dataset(file_location,lookback=60,lookforward=0,skips=0,data_sets=[45],eval_params=(mn,sd))
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
    test_set = sequential_dataset(file_location,lookback=60,lookforward=0,skips=0,data_sets=[test],eval_params=(mn,sd))
    test_loader = data.DataLoader(test_set,batch_size=64,shuffle = False)
    model_preds,model_labels = inference(test_loader,model,device,correction = (mn,sd))
    model_preds = model_preds[80:-10,:]
    model_labels = model_labels[80:-10,:]
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
