#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:12:55 2021

@author: charm
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import dataset_lib as dat
import torchvision.transforms as transforms
import torch.optim as opt
import torch.utils.data as torchdata
import time
import numpy as np
from tqdm import tqdm
import pdb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import models as mdl

class Encoder(nn.Module):
    
    def __init__(self,latent_dim):
        super(Encoder,self).__init__()
        self.latent_dim = latent_dim
        self.state_encoder = nn.Sequential(
                nn.Linear(54,512),
                nn.ReLU(),
                nn.Linear(512,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024,int(latent_dim)),
                nn.ReLU()
            )
        
        self.vision_encoder = torchvision.models.resnet50(pretrained=True)
        self.vision_encoder.fc = nn.Linear(2048,int(latent_dim))
        
        self.state_miu = nn.Linear(latent_dim,latent_dim)
        self.state_sig = nn.Linear(latent_dim,latent_dim)
        self.img_miu = nn.Linear(latent_dim,latent_dim)
        self.img_sig = nn.Linear(latent_dim,latent_dim)

    def mixture_of_experts(self,s_mean,s_logvar,i_mean,i_logvar):
        s_var = torch.clamp(s_logvar.exp(),1e-15,1e15)
        i_var = torch.clamp(i_logvar.exp(),1e-15,1e15)
        var = torch.clamp(1.0/(1.0/s_var +1.0/i_var),1e-15,1e15)
        mean = (s_mean/s_var+i_mean/i_var) * var
        
        return mean,var
                                  
    def forward(self,img,state):
        
        state_encoding = self.state_encoder(state)
        img_encoding = self.vision_encoder(img)
        state_mean = self.state_miu(state_encoding)
        state_logvar = self.state_sig(state_encoding)
        img_mean = self.img_miu(img_encoding)
        img_logvar = self.img_sig(img_encoding)
        
        mean,var = self.mixture_of_experts(state_mean,state_logvar,img_mean,img_logvar)
        
        return mean,var

class Decoder(nn.Module):

    def __init__(self,latent_size):
        super(Decoder,self).__init__()
        self.decoder = nn.Sequential(
                nn.Linear(latent_size,latent_size*2), # was 1024
                nn.BatchNorm1d(latent_size*2),
                nn.ReLU(),
                nn.Linear(latent_size*2,latent_size*2),
                nn.BatchNorm1d(latent_size*2),
                nn.ReLU(),
                nn.Linear(latent_size*2,latent_size),
                nn.ReLU(),
                nn.BatchNorm1d(latent_size),
                nn.Linear(latent_size,3))
        
    def forward(self,x):
        out = self.decoder(x)
        
        return out

class VAE(nn.Module):
    def __init__(self,latent_size):
        super(VAE,self).__init__()      
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)
    
    def forward(self,img,state):
        
        mean,var = self.encoder(img,state)
        z = draw_sample(mean,var)
        output = self.decoder(z)
        
        return output,mean,var
        
def draw_sample(mean,var):
    eps = torch.randn_like(mean)
    sigma = torch.sqrt(var)
    z = mean + (eps*sigma)
    return z

def loss_function(target,output,mean,var):
    
    mse = F.mse_loss(target,output,reduction="sum")
    KL = -0.5*torch.sum(1+torch.log(var)-mean.pow(2)-var)
    total_loss = mse+KL
    return total_loss

def KL_loss(mean,var,gamma=1.0):
    KL = -0.5*torch.sum(1+torch.log(var)-mean.pow(2)-var)
    return gamma*KL

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10, model_type = "VS", weight_file = "best_modelweights.dat", L1_loss = 0 ,suppress_log=False, hyperparam_search = False,lr_sched=None, multigpu=False, ablation=None,loss_dicts=None, beta_scheduling=None):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
            print("using GPU acceleration")
    if ablation is None:
        if multigpu and torch.cuda.device_count() > 1:
            print("multigpu enabled")
            model = nn.DataParallel(model)
            model = model.to(device,dtype=torch.float)
        else:
            model = model.to(device,dtype=torch.float)

    since = time.time()
    best_loss = np.Inf
    
    it_val = 0
    it_train = 0
    
    if beta_scheduling is None:
        beta_val = 0.001
    
    for epoch in range(num_epochs):
        if suppress_log==False:
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
        
        if beta_scheduling is not None:
            beta_val = beta_scheduling[epoch]
        
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
            running_loss1 = 0.0
            running_loss2 = 0.0

            # Iterate over data.
            batch_size = 0
            it = 1
            
            for inputs, aug_inputs, labels in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # here we do the feature dropout
                if ablation is not None:
                    drop = np.random.uniform()
                    if drop <= 0.2:
                        ablation = "V"
                        #print("ablating V")
                    elif drop > 0.2 and drop <= 0.4:
                        ablation = "S"
                        #print("ablating S")
                    else:
                        ablation = "N"
                
                if model_type !="S":
                    if ablation=="V": #ablate images and try to reconstruct them?
                        inputs = torch.zeros_like(inputs).to(device,dtype=torch.float)
                    else:
                        inputs = inputs.to(device,dtype=torch.float)
                
                if (model_type != "V") or (model_type!="V_RNN"):
                    if ablation=="S":
                        aug_inputs = torch.zeros_like(aug_inputs)
                        aug_inputs = aug_inputs.to(device,dtype=torch.float)
                    else:
                        aug_inputs = aug_inputs.to(device,dtype=torch.float)
                    
                labels = labels.to(device,dtype=torch.float)

                # forward
                # track history if only in train
                if phase == 'train':
                  torch.set_grad_enabled(True)
                  
                  if (model_type == "V") or (model_type=="V_RNN"):
                      outputs = model(inputs)
                  elif model_type == "VS":
                      #outputs = model(inputs,aug_inputs)
                      outputs,mean,var = model(inputs,aug_inputs)
                  else:
                      outputs = model(aug_inputs)
                  
                  loss1 = criterion(labels,outputs)
                  loss2 = KL_loss(mean,var,gamma=beta_val)
                  loss = loss1+loss2
                  
                  if L1_loss:
                      L1 = 0
                      for param in model.parameters():
                          if param.requires_grad:
                              L1 += L1_loss*torch.sum(torch.abs(param))
                      loss = loss+L1 
                  
                  if multigpu:
                      loss.mean().backward()
                  else:
                      loss.backward()
                  
                  optimizer.step()
                else :
                  torch.set_grad_enabled(False)
                  
                  if (model_type == "V") or (model_type=="V_RNN"):
                      outputs = model(inputs)
                  elif model_type == "VS":
                      outputs,mean,var= model(inputs,aug_inputs)
                  else:
                      outputs = model(aug_inputs)   
                  
                  loss1 = criterion(labels,outputs)
                  loss2 = KL_loss(mean,var,gamma=beta_val)
                  loss = loss1+loss2
                  #predictions = np.vstack((predictions,outputs.cpu().detach().numpy()))
                
                # statistics
                running_loss += loss.item() #* inputs.size(0) # multiply by the number of elements to get back the total loss, usually the loss function outputs the mean
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                batch_size += inputs.size(0)
                avg_loss = running_loss/batch_size
                avg_loss1 = running_loss1/batch_size
                avg_loss2 = running_loss2/batch_size
                loss_dicts[phase]['total'].append(avg_loss)
                loss_dicts[phase]['mse'].append(avg_loss1)
                loss_dicts[phase]['kl'].append(avg_loss2)
                
                if it%100 == 0 and suppress_log==False:
                  print('average loss for batch ' + str(it)+ ' : ' + str(avg_loss) + " MSE: " + str(avg_loss1) + " KL: "+str(avg_loss2))
                it +=1

            epoch_loss = running_loss / dataset_sizes[phase] #divide by the total size of our dataset to get the mean loss per instance
            epoch_loss1 = running_loss1/dataset_sizes[phase]
            epoch_loss2 = running_loss2/dataset_sizes[phase]
            
            if phase == 'val':
                if lr_sched is not None:
                    lr_sched.step(epoch_loss)
            
            if suppress_log==False:
                print('{} Loss: {:.4f} MSE: {:.4f} KL: {:.4f}'.format(phase, epoch_loss,epoch_loss1,epoch_loss2))
                if phase == "val":
                    print('Best Loss: {:.4f}'.format(best_loss))
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                
                if hyperparam_search==False:
                    print('Saving model... current loss:' + str(round(epoch_loss,5)) + ' < best loss: ' + str(round(best_loss,5)))
                    print("Backing up the model")
                    temp_file = open(weight_file,"wb")
                    if multigpu:
                        torch.save(model.module.state_dict(),temp_file)
                    else:
                        torch.save(model.state_dict(),temp_file)
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
        if multigpu:
            model.module.load_state_dict(torch.load(temp_file))
        else:
            model.load_state_dict(torch.load(temp_file))
    
    return model, best_loss  



#%%

if __name__ == "__main__":
    
    file_dir = '/home/charm/data_driven_force_estimation/experiment_data' # define the file directory for dataset
    
    model_type = "VS"
    feat_extract = False
    force_align = False
    
    weight_file =  weight_file = "best_modelweights_" + model_type
    
    
    trans_function = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    # We have to define the crop area of interest for the images
    # I hope to create cross-hairs in the gui so I can "aim" better during data collection.
    # That would help make the crop area consistent.
    
    crop_list = []
    
    for i in range(1,56+1):
        #crop_list.append((50,350,300,300))
        crop_list.append((270-150,480-150,300,300))
      
    train_list = [1,3,5,7,
                  8,10,12,14,
                  15,17,19,21,41,42]
    val_list = [2,6,
                9,13,
                16,20,44]
    '''
    test_list = [4]
    config_dict={'file_dir':file_dir,
                 'include_torque': False,
                 'spatial_forces': force_align,
                 'custom_state': None,
                 'batch_size': 32,
                 'crop_list': crop_list,
                 'trans_function': trans_function}
    
    dataloaders,dataset_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict,augment=False)
    '''
    
    # to have more control over the dataloading we declare our own loaders
    dataset_train = dat.ImgDataset(file_dir, file_dir,data_sets=train_list,transform=trans_function,crop_list=crop_list)
    dataset_val = dat.ImgDataset(file_dir, file_dir,data_sets=val_list,transform=trans_function,crop_list=crop_list,eval_params=(dataset_train.mean,dataset_train.stdev))
    dataloaders = {}
    dataset_sizes={}
    dataloaders['train'] = torchdata.DataLoader(dataset_train,batch_size=32,shuffle=True,num_workers=4)
    dataloaders['val'] = torchdata.DataLoader(dataset_val,batch_size=32,shuffle=False,num_workers=4)   
    dataset_sizes['train'] = len(dataset_train)
    dataset_sizes['val'] = len(dataset_val)
    
    #%%
    #generate_grid(dataloaders['test'].dataset,64)

    # define model
    z_dim = 256
    model = VAE(z_dim)
    
    weight_file = weight_file + "_VAE.dat"
    # create loss function
    criterion = nn.MSELoss(reduction='sum')
    # define optimization method
    optimizer = opt.Adam(model.parameters(),lr=1e-3,weight_decay=0)
    #optimizer = opt.SGD(model.parameters(),lr=1e-5,weight_decay=0,momentum=0.9)
    lrsched = opt.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=10,threshold = 0.0001,verbose=True)
    loss_dicts = {}
    train_stages= ["train","val"]
    loss_types = ["total","mse","kl"]
    for stage in train_stages:
        loss_dicts[stage]={}
        for loss_type in loss_types:
            loss_dicts[stage][loss_type]=[]
    
    beta_sched= np.linspace(0,1,30)
    
    model,_ = train_model(model,
                            criterion, optimizer,
                            dataloaders, dataset_sizes,  
                            num_epochs=30,
                            L1_loss=0,
                            model_type= model_type,
                            weight_file=weight_file,
                            suppress_log=False,
                            multigpu=True,
                            loss_dicts=loss_dicts,
                            lr_sched=None,
                            beta_scheduling=beta_sched
                            )
   
#%% 
    '''
    test_list =  [4,11,18,
              22,23,
              24,25,
              26,27,
              28,29,
              32,33,
              34,36,
              37,38,39,
              45,46,47]
    '''
    
    #model = VAE(z_dim)
    #model.load_state_dict(torch.load(weight_file))
    
    #model = mdl.StateVisionModel(30, 54, 3,feature_extract=False)
    #model.load_state_dict(torch.load("031421_experiments/best_modelweights_VS.dat"))
    #model = mdl.VisionModel(3)
    #model.load_state_dict(torch.load("031421_experiments/best_modelweights_V.dat"))
    val_list = [2]
    dataset_val = dat.ImgDataset(file_dir, file_dir,data_sets=val_list,transform=trans_function,crop_list=crop_list,eval_params=(dataset_train.mean,dataset_train.stdev))
    dataloaders['val'] = torchdata.DataLoader(dataset_val,batch_size=32,shuffle=False,num_workers=4)   
    model =model.to("cuda")
    pred = np.zeros((0,3))
    labels = np.zeros((0,3))
    model.eval()
    for img,state,label in dataloaders['val']:
        img = img.to("cuda",dtype=torch.float)
        state = state.to("cuda",dtype=torch.float)
        with torch.no_grad():
            pred_out,mn,vr = model(img,state)
            #pred_out =model(img,state)
            #pred_out = model(img)
        pred = np.vstack((pred,pred_out.cpu().numpy()))
        labels = np.vstack((labels,label.numpy()))
        
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(3,1)
    for i in range(3):
        ax[i].plot(pred[:,i])
        ax[i].plot(labels[:,i])
    
    plt.figure()
    for i in range(1,3):
        metric = loss_types[i]
        plt.plot(loss_dicts['train'][metric])
        #plt.plot(loss_dicts['val'][metric])
        
        
#%%
        
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

condition_list = ['center','right','left',
                  'right_less','right_less',
                  'right_more','right_more',
                  'left_less','left_less',
                  'left_more','left_more',
                  'new_tool','new_tool',
                  'new_material','new_material',
                  'center','right','left',
                  'z_mid','z_high','z_low']

test_list = [4,11,18,
            22,23,
            24,25,
            26,27,
            28,29,
            32,33,
            34,36,
            37,38,39,
            45,46,47]

train_list = [1,3,5,7,
              8,10,12,14,
              15,17,19,21,41,42]
val_list = [2,6,
            9,13,
            16,20,44]

'''
test_list = [4]
config_dict={'file_dir':file_dir,
             'include_torque': False,
             'spatial_forces': force_align,
             'custom_state': None,
             'batch_size': 32,
             'crop_list': crop_list,
             'trans_function': trans_function}

dataloaders,dataset_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict,augment=False)
'''

import torch.utils.data as data

# to have more control over the dataloading we declare our own loaders
dataset_train = dat.ImgDataset(file_dir, file_dir,data_sets=train_list,transform=trans_function,crop_list=crop_list)
dataset_val = dat.ImgDataset(file_dir, file_dir,data_sets=val_list,transform=trans_function,crop_list=crop_list,eval_params=(dataset_train.mean,dataset_train.stdev))
dataloaders = {}
dataset_sizes={}
#dataloaders['train'] = torchdata.DataLoader(dataset_train,batch_size=32,shuffle=True)
#dataloaders['val'] = torchdata.DataLoader(dataset_val,batch_size=32,shuffle=False)   
#dataset_sizes['train'] = len(dataset_train)
#dataset_sizes['val'] = len(dataset_val)

max_force = []
min_force = []
for test in test_list:
    forces = np.loadtxt(file_dir+'/labels_'+str(test)+'.txt',delimiter = ",")[:,1:4]
    test_max = np.max(forces,axis=0)
    test_min = np.min(forces,axis=0)
    max_force.append(test_max)
    min_force.append(test_min) 

mn = dataset_train.mean[1:4]
sd = dataset_train.stdev[1:4]
pred_list = []
metrics_list = []
#model = VAE(56)
#model.load_state_dict(torch.load("best_modelweights_VS_VAE_56.dat"))
model = nn.DataParallel(model)
model = model.to("cuda")
model.eval()
for i,test in enumerate(test_list):
    #test_set = EncodedDataset(file_dir, encode_dir, data_sets = [test], eval_params = (train_set.mean,train_set.stdev))
    test_set = dat.ImgDataset(file_dir,file_dir,
                           data_sets=[test],
                           transform = trans_function,
                           crop_list=crop_list,
                           eval_params = (dataset_train.mean,dataset_train.stdev),
                           include_torque= False,
                           custom_state= None,
                           color_jitter=False,
                           crop_jitter=False,
                           rand_erase=False)
    test_loader = data.DataLoader(test_set,batch_size=32,shuffle=False)
    preds = np.empty((0,3))
    for img,x,y in test_loader:
        with torch.no_grad():
            img = img.to("cuda:0",dtype=torch.float)
            x = x.to("cuda:0",dtype=torch.float)
            pred,_,_ = model(img,x)
            #pred = model(x)
        preds = np.vstack((preds,(pred.cpu().numpy()*sd)+mn))
    pred_list.append(preds)
    condition=condition_list[i]
    print(condition)
    metrics = compute_loss_metrics(preds[10:-10,:],test_loader.dataset.raw_label_array[10:-10,1:4],max_force[i],min_force[i],condition,"ENC")
    metrics_list.append(metrics)
    
import pickle    

pickle.dump(metrics_list,open("df_VAE56_test.df","wb"))
pickle.dump(pred_list,open("preds_VAE56.pred","wb"))