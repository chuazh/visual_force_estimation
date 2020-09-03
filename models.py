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

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10, model_type = "VS", weight_file = "best_modelweights.dat"):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("using GPU acceleration")
    
    model = model.to(device,dtype=torch.float)
    
    since = time.time()
    best_loss = np.Inf
    
    train_losses = np.zeros(num_epochs*dataset_sizes['train'])
    val_losses = np.zeros(num_epochs*dataset_sizes['val'])
    
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

            for inputs, aug_inputs, labels in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()
                
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
                  loss.backward()
                  #xm.optimizer_step(optimizer,barrier=True)
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
                
                # statistics
                running_loss += loss.item() #* inputs.size(0) # multiply by the number of elements to get back the total loss, usually the loss function outputs the mean
                batch_size += inputs.size(0)
                avg_loss = running_loss/batch_size
                
                if phase== 'train':
                    train_losses[it] = avg_loss
                else:
                    val_losses[it] = avg_loss
                
                if it%10 == 0:
                  print('average loss for iteration ' + str(it)+ ' : ' + str(avg_loss))
                it += 1

            epoch_loss = running_loss / dataset_sizes[phase] #divide by the total size of our dataset to get the mean loss per instance
            
            
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print('Saving model... current loss:' + str(round(epoch_loss,5)) + ' < best loss: ' + str(round(best_loss,5)))
                best_loss = epoch_loss
                print("Backing up the model")
                temp_file = open(weight_file,"wb")
                torch.save(model.state_dict(),temp_file)

                
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    temp_file.close()
    temp_file = open(weight_file,"rb")
    model.load_state_dict(torch.load(temp_file))
    return model, train_losses, val_losses     