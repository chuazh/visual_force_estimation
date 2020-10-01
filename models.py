#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:46:51 2020

@author: charm

Model specification and Training Code

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import time
import numpy as np
from tqdm import tqdm
import pdb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except:
    pass

class StateVisionModel(nn.Module):
  
  '''
  A state + vision model
  cnn_out: the number of output states from the linear layer of the ResNet50 Model
  augmented_in: the number of features from the state input vector
  model_out: the number of output dimensions
  '''
  
  def __init__(self,cnn_out,augmented_in,model_out,feature_extract=False,layer_depth=4):
    super(StateVisionModel,self).__init__()
    self.cnn = models.resnet50(pretrained=True)

    layer_num = 0
    for param in self.cnn.parameters(): # this sets up fine tuning of the residual layers
      if feature_extract:
          param.requires_grad=False
      else:
          if layer_num < layer_depth :
              param.requires_grad = False
              layer_num += 1
    
    self.cnn.fc = nn.Linear(self.cnn.fc.in_features,cnn_out) # the fully connected layer will compress the output into 30 params
    # create a few more layers to take use through the data
    self.fc1 = nn.Linear(cnn_out+augmented_in,180)
    self.fc2 = nn.Linear(180,50)
    self.fc3 = nn.Linear(50,model_out)

    # create a batchnorm layer 
    self.bn1 = nn.BatchNorm1d(num_features=180)
    self.bn2 = nn.BatchNorm1d(num_features=50)

  def forward(self,image,data):
    x1 = self.cnn(image)
    x2 = data

    x = torch.cat((x1,x2),dim=1)
    x = F.relu(self.bn1(self.fc1(x)))
    x = F.relu(self.bn2(self.fc2(x)))
    x = self.fc3(x)

    return x

def VisionModel(output_dim,layer_depth=4):
    
    model = models.resnet50(pretrained=True)
    layer_num = 0
    
    for param in model.parameters():
        if layer_num < layer_depth:
            param.requires_grad = False
            layer_num = layer_num + 1

    num_features = model.fc.in_features # get the input size to the FC layer
    model.fc = nn.Linear(num_features,output_dim) # directly map it to the 
    
    #state_dict = torch.load('../ML dvrk 072820/train124_val35_modelweights.dat')
    #model_ft.load_state_dict(state_dict)
    
    return model


def StateModel(input_dim,output_dim):
    
    model = nn.Sequential(
    nn.Linear(input_dim,500),
    nn.BatchNorm1d(500),
    nn.ReLU(),
    nn.Linear(500,1000),
    nn.BatchNorm1d(1000),
    nn.ReLU(),
    nn.Linear(1000,1000),
    nn.BatchNorm1d(1000),
    nn.ReLU(),
    nn.Linear(1000,1000),
    nn.BatchNorm1d(1000),
    nn.ReLU(),
    nn.Linear(1000,500),
    nn.BatchNorm1d(500),
    nn.ReLU(),
    nn.Linear(500,50),
    nn.BatchNorm1d(50),
    nn.ReLU(),
    nn.Linear(50,output_dim)
    )
    
    return model

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10, model_type = "VS", weight_file = "best_modelweights.dat", L1_loss = 0 ,suppress_log=False, hyperparam_search = False, use_tpu=False, tensorboard = True):
    
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
    train_losses = np.zeros(num_epochs*len(dataloaders['train']))
    val_losses = np.zeros(num_epochs*len(dataloaders['val']))
    
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
                if dataloaders[phase].dataset.include_torque:
                    predictions = np.empty((0,6))
                else:
                    predictions = np.empty((0,3))

            running_loss = 0.0

            # Iterate over data.
            batch_size = 0
            it = 1
            
            for inputs, aug_inputs, labels in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()
                
                if model_type !="S":
                    inputs = inputs.to(device,dtype=torch.float)
                
                if model_type != "V":
                    aug_inputs = aug_inputs.to(device,dtype=torch.float)
                    
                labels = labels.to(device,dtype=torch.float)

                # forward
                # track history if only in train
                if phase == 'train':
                  torch.set_grad_enabled(True)
                  
                  if model_type == "V":
                      outputs = model(inputs)
                  elif model_type == "VS":
                      outputs= model(inputs,aug_inputs)
                  else:
                      outputs = model(aug_inputs)
                      
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
                  
                  if model_type == "V":
                      outputs = model(inputs)
                  elif model_type == "VS":
                      outputs= model(inputs,aug_inputs)
                  else:
                      outputs = model(aug_inputs)
                      
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
                    temp_file = open(weight_file,"wb")
                    torch.save(model.state_dict(),temp_file)
                    if tensorboard:
                        fig,ax = plt.subplots(3,1,sharex=True,figsize=(50,10))
                        plt.ioff()
                        for f_ax in range(3):
                            ax[f_ax].plot(dataloaders[phase].dataset.label_array[:,f_ax+1])
                            ax[f_ax].plot(predictions[:,f_ax],linewidth=1)
                        writer.add_figure('valPred/figure',fig,global_step=epoch,close=True)
                
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
        temp_file = open(weight_file,"rb")
        model.load_state_dict(torch.load(temp_file))
    
    return model, train_losses, val_losses, best_loss     


def evaluate_model(model,dataloader,model_type="S",no_pbar=False):
    
    tqdm.write('Performing Inference...')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        tqdm.write("using GPU acceleration")
    
    model = model.to(device,dtype=torch.float)
    model.eval()
    
    if dataloader.dataset.include_torque == False:
        predictions = np.empty((0,3))
    else:
        predictions = np.empty((0,6))
    
    with tqdm(total=len(dataloader),leave=True,miniters=1,disable=no_pbar) as pbar:
        for inputs, aug_inputs, labels in dataloader:
            
            if model_type !="S":
                inputs = inputs.to(device,dtype=torch.float)
            if model_type != "V":
                aug_inputs = aug_inputs.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            
            if model_type == "V":
                outputs = model(inputs)
            elif model_type == "VS":
                outputs= model(inputs,aug_inputs)
            else:
                outputs = model(aug_inputs)
                          
            predictions = np.vstack((predictions,outputs.cpu().detach().numpy()))
            pbar.update(1)
            
    return predictions
            
