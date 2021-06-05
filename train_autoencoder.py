#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 21:41:40 2021

@author: charm
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import dataset_lib as dat
import torchvision.transforms as transforms
import torch.optim as opt
import time
import numpy as np
from tqdm import tqdm
import pdb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


unnorm = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

class AutoEncoder(nn.Module):
    def __init__(self,latent_size):
        super(AutoEncoder,self).__init__()
        self.encoder = torchvision.models.resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(2048,latent_size)
        self.decoder = Decoder(latent_size)
        
    def forward(self,x):
        x = self.encoder(x)
        x = F.relu(x)
        x = self.decoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self,latent_size):
        super(Decoder,self).__init__()
        
        self.layer_size = 7
        self.channels = 512
        self.fc2img = nn.Linear(latent_size,self.channels*self.layer_size*self.layer_size)
        self.bn= nn.BatchNorm1d(self.channels*self.layer_size*self.layer_size)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,2,2),
            nn.Sigmoid())
        
    def forward(self,x):
		
        x = self.fc2img(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(x.shape[0],self.channels,self.layer_size,self.layer_size)
        x = self.decode(x)

        return x

class state_autoencoder(nn.Module):
    
    def __init__(self,latent_size):
        super(state_autoencoder,self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(54,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,int(latent_size)),
            nn.ReLU()
        )
        
        self.state_decoder = nn.Sequential(
                nn.Linear(latent_size,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.Linear(512,54))
    
    def forward(self,x):
        x = self.state_encoder(x)
        x = self.state_decoder(x)
        
        return x
        
class VS_autoencoder(nn.Module):
    
    def __init__(self,latent_size):
        super(VS_autoencoder,self).__init__()
        self.state_encoder = nn.Sequential(
                nn.Linear(54,512),
                nn.ReLU(),
                nn.Linear(512,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024,int(latent_size/2)),
                nn.ReLU()
            )
        self.vision_encoder = torchvision.models.resnet50(pretrained=True)
        self.vision_encoder.fc = nn.Linear(2048,int(latent_size/2))
        self.bottleneck = nn.Sequential(
            nn.Linear(latent_size,latent_size),
            nn.ReLU(),
            nn.Linear(latent_size,latent_size),
            nn.ReLU())
        self.vision_decoder = Decoder(latent_size)
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_size,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,54))  
        
    def forward(self,x,bypass=False):
        v,s = x
        v_latent = self.vision_encoder(v)
        s_latent = self.state_encoder(s)
        full_latent = torch.cat((v_latent,s_latent),dim=1)
        full_out = self.bottleneck(full_latent)
        if not bypass:
            v_out = self.vision_decoder(full_latent)
            s_out = self.state_decoder(full_latent)
        else:
            v_out = None
            s_out = None
        
        return v_out,s_out,full_out

class latent_forcedecoder2(nn.Module):

    def __init__(self):
        super(latent_forcedecoder2,self).__init__()
        self.decoder = nn.Sequential(
                nn.Linear(1024,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512,3))
        
    def forward(self,x):
        out = self.decoder(x)
        
        return out

class full_forcedecoder(nn.Module):
    
    def __init__(self):
        super(full_forcedecoder,self).__init__()
        self.encoder = VS_autoencoder(1024)
        self.decoder = latent_forcedecoder2()
        self.encoder.load_state_dict(torch.load("/home/charm/data_driven_force_estimation/clean_code/031421_experiments/best_modelweights_VS_autoencode_dropav_notpara.dat"))

    def forward(self,img,state):
        _,_,enc = self.encoder((img,state),True)
        out = self.decoder(enc)
        
        return out


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10, model_type = "VS", weight_file = "best_modelweights.dat", L1_loss = 0 ,suppress_log=False, hyperparam_search = False, use_tpu=False, multigpu=False,tensorboard = True, ablation=None):
    
    if use_tpu:
        print("using TPU acceleration, model and optimizer should already be loaded onto tpu device")
        device = xm.xla_device()
    else:
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
                    
                #labels = labels.to(device,dtype=torch.float)

                # forward
                # track history if only in train
                if phase == 'train':
                  torch.set_grad_enabled(True)
                  
                  if (model_type == "V") or (model_type=="V_RNN"):
                      outputs = model(inputs)
                  elif model_type == "VS":
                      #outputs = model(inputs,aug_inputs)
                      outputs = model((inputs,aug_inputs))
                  else:
                      outputs = model(aug_inputs)
                  
                  if model_type!="S":
                      for img_num in range(inputs.shape[0]):
                          inputs[img_num,:,:,:] = unnorm(inputs[img_num,:,:,:])
                     
                  if model_type=="V":      
                      loss = criterion(outputs,inputs)
                  elif model_type == "VS":
                      loss1 = 100*criterion(outputs[0],inputs)
                      loss2 = criterion(outputs[1],aug_inputs)
                      loss = loss1+loss2
                  elif model_type == "S":
                      loss = criterion(outputs,aug_inputs)
                  
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
                  if use_tpu:
                      xm.optimizer_step(optimizer,barrier=True)
                  else:
                      optimizer.step()
                else :
                  torch.set_grad_enabled(False)
                  
                  if (model_type == "V") or (model_type=="V_RNN"):
                      outputs = model(inputs)
                  elif model_type == "VS":
                      outputs= model((inputs,aug_inputs))
                  else:
                      outputs = model(aug_inputs)
                  if model_type!="S":
                      for img_num in range(inputs.shape[0]):
                          inputs[img_num,:,:,:] = unnorm(inputs[img_num,:,:,:])    
                  
                  if model_type=="V":      
                      loss = criterion(outputs,inputs)
                  elif model_type == "VS":
                      loss1 = criterion(outputs[0],inputs)
                      loss2 = criterion(outputs[1],aug_inputs)
                      loss = 100*loss1+loss2
                  elif model_type=="S":
                      loss = criterion(outputs,aug_inputs)
                  #predictions = np.vstack((predictions,outputs.cpu().detach().numpy()))
                
                # statistics
                running_loss += loss.item() #* inputs.size(0) # multiply by the number of elements to get back the total loss, usually the loss function outputs the mean
                if model_type=="VS":
                    running_loss1 += loss1.item()
                    running_loss2 += loss2.item()
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
                  print('average loss for batch ' + str(it)+ ' : ' + str(avg_loss) + "img: " + str(running_loss1/batch_size) + " state: " + str(running_loss2/batch_size))
                it +=1

            epoch_loss = running_loss / dataset_sizes[phase] #divide by the total size of our dataset to get the mean loss per instance
            if model_type =="VS":
                epoch_loss1 = running_loss1/dataset_sizes[phase]
                epoch_loss2 = running_loss2/dataset_sizes[phase]
            else:
                epoch_loss1 = 0.0
                epoch_loss2 = 0.0
            if tensorboard:
                if phase=="train":
                    writer.add_scalar('ELoss/train',epoch_loss,epoch)
                if phase=="val":
                     writer.add_scalar('ELoss/val',epoch_loss,epoch)
                 
            
            if suppress_log==False:
                print('{} Loss: {:.4f} img:{:.4f} st:{:.4f} '.format(phase, epoch_loss,epoch_loss1,epoch_loss2))
            
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

#%%

if __name__ == "__main__":
    
    file_dir = '/home/charm/data_driven_force_estimation/experiment_data' # define the file directory for dataset
    
    model_type = "VS"
    ablate = "S"
    
    feat_extract = False
    force_align = False
    
    weight_file =  weight_file = "best_modelweights_" + model_type
    
    if model_type!="S" and feat_extract:
        weight_file="best_modelweights_" + model_type + "_ft"
        
    if force_align and model_type!= "V" :
        weight_file = weight_file + "_faligned"
        
    if model_type == "V_RNN":
        trans_function = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        # Define a transformation for the images
        trans_function = transforms.Compose([transforms.Resize((224,224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    # We have to define the crop area of interest for the images
    # I hope to create cross-hairs in the gui so I can "aim" better during data collection.
    # That would help make the crop area consistent.
    
    crop_list = []
    
    for i in range(1,48):
        #crop_list.append((50,350,300,300))
        crop_list.append((270-150,480-150,300,300))
        
    train_list = [1,3,5,7,
                  8,10,12,14,
                  15,17,19,21,41,42]
    val_list = [2,6,
                9,13,
                16,20,44]
    test_list = [1]
    #train_list = [1,3,5,7,43,45]
    #val_list = [2,6,44,46]
    #test_list = [4,11,18,
                 #22,23,24,25,26,27,28,29,32,33]
    #test_list = [4,8]
    
    config_dict={'file_dir':file_dir,
                 'include_torque': False,
                 'spatial_forces': force_align,
                 'custom_state': None,
                 'batch_size': 32,
                 'crop_list': crop_list,
                 'trans_function': trans_function}
    
    dataloaders,dataset_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict,augment=False)
    
    #end of ablation code
    
    #generate_grid(dataloaders['test'].dataset,64)

    # define model
    if model_type == "VS":
        #model = mdl.StateVisionModel(30, 54, 3,feature_extract=feat_extract,TFN=True)
        model = VS_autoencoder(1024)
        #model = full_forcedecoder()
    elif model_type == "S":
        model  = state_autoencoder(512)
    elif (model_type == "V") or (model_type == "V_RNN"):
        model = AutoEncoder(1024)
    
    weight_file = weight_file + "_autoencoder.dat"
    
    if ablate is not None:
        model = nn.DataParallel(model)
        model = model.to("cuda",dtype=torch.float)
        #model.load_state_dict(torch.load("best_modelweights_VS_autoencode_full.dat",map_location='cuda'),strict=False)
    
    # create loss function
    criterion = nn.MSELoss(reduction='mean')
    # define optimization method
    if ablate is None:
        optimizer = opt.Adam(model.parameters(),lr=1e-3,weight_decay=0)
    else:
        optimizer = opt.Adam(model.parameters(),lr=1e-3,weight_decay=0)
#%%
    model,train_history,val_history,_ = train_model(model,
                                                    criterion, optimizer,
                                                    dataloaders, dataset_sizes,  
                                                    num_epochs=100,
                                                    L1_loss=0,
                                                    model_type= model_type,
                                                    weight_file=weight_file,
                                                    suppress_log=False,
                                                    multigpu=True,
                                                    ablation=ablate)
    
#%% Visualize autoencoder result
    
if model_type=="VS":
    model = VS_autoencoder(1024)
    wts = torch.load("best_modelweights_VS_autoencode_dropav.dat",map_location='cuda:0')    
elif model_type =="V":
    model = AutoEncoder(1024)
    wts = torch.load("best_modelweights_V_autoencode.dat",map_location='cuda:0') 
elif model_type == "S":
    model = state_autoencoder(512)
    wts = torch.load("best_modelweights_S_autoencode.dat",map_location='cuda:0') 
    
model = nn.DataParallel(model)
model = model.to("cuda:0")
model.load_state_dict(wts,strict=False)
model.eval()
#%%
import matplotlib.pyplot as plt
import numpy as np

def imshow(img,orig=True):
    img = img #/ 2 + 0.5     # unnormalize
    if orig:
        img = unnorm(img)
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

from torch.utils import data
# we need to generate a random batchs
loader = data.DataLoader(dataloaders['train'].dataset,batch_size = 32, shuffle=True)

ablate="V"
if model_type=="V":    
    val_data = next(iter(loader))[0]
    with torch.no_grad():
        ae_output = model(val_data)
elif model_type=="VS":
    val_data = next(iter(loader))
    inputs = val_data[0].to("cuda:0",dtype=torch.float)
    aug_inputs = val_data[1].to("cuda:0",dtype=torch.float)
    if ablate == "S" or ablate =="VS":
        aug_inputs = torch.zeros_like(aug_inputs)
    if ablate == "V" or ablate=="VS":
        inputs = torch.zeros_like(inputs)
    with torch.no_grad():
        ae_output,ae_state,latent_out = model((inputs,aug_inputs))
elif model_type =="S":
    val_data = next(iter(loader))
    aug_inputs = val_data[1].to("cuda:0",dtype=torch.float)
    with torch.no_grad():
        ae_state = model(inputs)
#%%    
plt.close('all')
plt.figure()
imshow(torchvision.utils.make_grid(val_data[0]))
plt.figure()
imshow(torchvision.utils.make_grid(ae_output.cpu()),False)
#%%
qty = ['px','py','pz','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz',
       'q1','q2','q3','q4','q5','q6','q7',
       'vq1','vq2','vq3','vq4','vq5','vq6','vq7',
       'tq1','tq2','tq3','tq4','tq5','tq6','tq7',
       'q1d','q2d','q3d','q4d','q5d','q6d','q7d',
       'tq1d','tq2d','tq3d','tq4d','tq5d','tq6d','tq7d',
       'psm_fx','psm_fy','psm_fz','psm_tx','psm_ty','psm_tz']

plt.close('all')
fig,ax = plt.subplots(3)
for n in range(len(ax)):
    ax[n].plot(val_data[1].numpy()[:,n])
    ax[n].plot(ae_state.cpu().numpy()[:,n])
fig.suptitle("xyz pos")

fig,ax = plt.subplots(3)
for n in range(len(ax)):
    ax[n].plot(val_data[1].numpy()[:,n+48])
    ax[n].plot(ae_state.cpu().numpy()[:,n+48])

fig,ax = plt.subplots(7)
for n in range(len(ax)):
    ax[n].plot(val_data[1].numpy()[:,n+13])
    ax[n].plot(ae_state.cpu().numpy()[:,n+13])

#%% To speed up the training of a decoder we can upfront just compute encoded values
from torch.utils import data

train_list = [1,3,5,7,
              8,10,12,14,
              15,17,19,21,41,42]

#test_list = [1,3,5,7,8,10,12,14,15,17,19,21,41,42,
             #2,6,9,13,16,20,44,4,11,18,22,23,24,25,26,27,28,29,32,33,34,36,37,38,39,45,46,47]

trans_function = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
crop_list = []
    
for i in range(1,48):
    #crop_list.append((50,350,300,300))
    crop_list.append((270-150,480-150,300,300))

# init the train_set
train_set = dat.ImgDataset(file_dir,file_dir,
                           data_sets=train_list,
                           transform = trans_function,
                           crop_list=crop_list,
                           include_torque= False,
                           custom_state= None,
                           color_jitter=False,
                           crop_jitter=False,
                           rand_erase=False)

# encode data in for loop
for i in range(1,48):
    model.eval()
    test_set = dat.ImgDataset(file_dir,file_dir,
                          data_sets=[i],
                          transform = trans_function,
                          crop_list=crop_list,
                          include_torque= False,
                          eval_params=(train_set.mean,train_set.stdev),
                          custom_state= None,
                          color_jitter=False,
                          crop_jitter=False,
                          rand_erase=False)
    
    loader = data.DataLoader(test_set,batch_size = 32, shuffle=False)
    
    # create new file
    print("creating file: encodings_" +  str(i)+ ".enc")
    f = open('encodings_'+str(i)+'.enc','w')
    f.close()
    f = open('encodings_'+str(i)+'.enc','a') # open in append mode
    print("writing...")
    for inputs, aug_inputs, labels in loader:
        with torch.no_grad():
            inputs = inputs.to("cuda:0",dtype=torch.float)
            aug_inputs = aug_inputs.to("cuda:0",dtype=torch.float)
            img_out,state_out,latent_out = model((inputs,aug_inputs))
        np.savetxt(f,latent_out.cpu().numpy())
        f.write("\n")

#%%
from torch.utils import data   
import natsort
import glob

class latent_forcedecoder(nn.Module):

    def __init__(self):
        super(latent_forcedecoder,self).__init__()
        self.decoder = nn.Sequential(
                nn.Linear(1024,180),
                nn.BatchNorm1d(180),
                nn.ReLU(),
                nn.Linear(180,50),
                nn.BatchNorm1d(50),
                nn.ReLU(),
                nn.Linear(50,3))
        
    def forward(self,x):
        out = self.decoder(x)
        
        return out
 
class latent_forcedecoder2(nn.Module):

    def __init__(self):
        super(latent_forcedecoder2,self).__init__()
        self.decoder = nn.Sequential(
                nn.Linear(1024,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512,3))
        
    def forward(self,x):
        out = self.decoder(x)
        
        return out
    
class EncodedDataset(data.Dataset):
    
    '''Characterizes a dataset for PyTorch'''
    def __init__(self, label_dir, encode_dir, data_sets = None, eval_params = None):
        '''
        Initialization
           exclude_index is a list denoting which datasets to exclude indexed
           from 1
        '''
        self.label_dir = label_dir
        self.encode_dir = encode_dir
        self.label_array = self.read_labels(self.label_dir,data_sets)
        self.encode_array = self.read_encodings(self.encode_dir,data_sets)
        if eval_params is None:
            self.mean,self.stdev = self.normalize_state()
        else:
            mean,std = eval_params
            self.mean,self.stdev = self.normalize_state(mean=mean,stdev=std)
            
        # extra legacy crap
        self.include_torque=False
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_array)

    def __getitem__(self, index):
        'Generates one sample of data'
        #x = self.label_array[index][7:61]
        x = self.encode_array[index,:]
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
                label_list.append(data)
        else:
            for i in range(len(file_list)):
                data = np.loadtxt(file_list[i],delimiter=",")
                label_list.append(data)
        
        labels = np.concatenate(label_list,axis=0)
        
        return labels
    
    def read_encodings(self,encode_dir,data_sets):
        file_list = natsort.humansorted(glob.glob(encode_dir + "/encodings_*.enc"))
        encode_list = []        
        if data_sets is not None:
            for i in data_sets:
                print(file_list[i-1])
                data = np.loadtxt(file_list[i-1])
                encode_list.append(data)
        else:
            for i in range(len(file_list)):
                data = np.loadtxt(file_list[i])
                encode_list.append(data)
        
        encoded = np.concatenate(encode_list,axis=0)
        
        return encoded
    
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

#%%
import models as mdl

file_dir = '/home/charm/data_driven_force_estimation/experiment_data' # define the file directory for dataset
encode_dir = '/home/charm/data_driven_force_estimation/clean_code/031421_experiments/autoencoded_full'

train_list = [1,3,5,7,
              8,10,12,14,
              15,17,19,21,41,42]
val_list = [2,6,
            9,13,
            16,20,44]

train_set = EncodedDataset(file_dir, encode_dir, data_sets = train_list, eval_params = None)
val_set = EncodedDataset(file_dir, encode_dir, data_sets = val_list, eval_params = (train_set.mean,train_set.stdev))


dataloader_dict = {}
dataloader_dict['train'] = data.DataLoader(train_set,batch_size = 32, shuffle=True)
dataloader_dict['val'] = data.DataLoader(val_set,batch_size = 32, shuffle=False)
dataset_sizes = {'train':len(train_set),'val':len(val_set)}
#%%
file_dir = '/home/charm/data_driven_force_estimation/experiment_data' # define the file directory for dataset
trans_function = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
# We have to define the crop area of interest for the images
# I hope to create cross-hairs in the gui so I can "aim" better during data collection.
# That would help make the crop area consistent.
model_type="VS"
crop_list = []

for i in range(1,48):
    #crop_list.append((50,350,300,300))
    crop_list.append((270-150,480-150,300,300))
    
train_list = [1,3,5,7,
              8,10,12,14,
              15,17,19,21,41,42]
val_list = [2,6,
            9,13,
            16,20,44]
test_list = [1]
#train_list = [1,3,5,7,43,45]
#val_list = [2,6,44,46]
#test_list = [4,11,18,
             #22,23,24,25,26,27,28,29,32,33]
#test_list = [4,8]

config_dict={'file_dir':file_dir,
             'include_torque': False,
             'spatial_forces': False,
             'custom_state': None,
             'batch_size': 32,
             'crop_list': crop_list,
             'trans_function': trans_function}

dataloader_dict,dataset_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict,augment=False)
#%%

#model = latent_forcedecoder2()
model = full_forcedecoder()
criterion = nn.MSELoss(reduction='sum')
optimizer = opt.Adam(model.parameters(),lr=1e-5,weight_decay=0)
#optimizer = opt.SGD(model.parameters(),lr=1e-5,weight_decay=0, momentum=0.9)
weight_file = "best_modelweights_ENC_fulldec.dat"

model,train_history,val_history,_ = mdl.train_model(model,
                                                     criterion, optimizer,
                                                     dataloader_dict, dataset_sizes,  
                                                     num_epochs=100,
                                                     L1_loss=0,
                                                     model_type= "VS",
                                                     weight_file=weight_file,
                                                     suppress_log=False,
                                                     multigpu=True,
                                                     tensorboard=False)

#%%
weight_file = "best_modelweights_ENC_large2.dat"
model = latent_forcedecoder2()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(weight_file))
model = model.to("cuda:0",dtype=torch.float)
model.eval()
#%%

test_list = [4]

test_set = EncodedDataset(file_dir, encode_dir, data_sets = test_list, eval_params = (train_set.mean,train_set.stdev))
test_loader = data.DataLoader(test_set,batch_size=32,shuffle=False)
pred_list = np.empty((0,3))
label_list = np.empty((0,3))

for img,x,y in test_loader:
    with torch.no_grad():
        x = x.to("cuda:0",dtype=torch.float)
        pred = model(x)
    pred_list=np.vstack((pred_list,pred.cpu().numpy()))
    label_list=np.vstack((label_list,y))
    
#%%
    
import matplotlib.pyplot as plt

fig,ax = plt.subplots(3,1)
for i in range(3):
    ax[i].plot(pred_list[:,i])
    ax[i].plot(label_list[:,i])
    
    
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

max_force = []
min_force = []
for test in test_list:
    forces = np.loadtxt(file_dir+'/labels_'+str(test)+'.txt',delimiter = ",")[:,1:4]
    test_max = np.max(forces,axis=0)
    test_min = np.min(forces,axis=0)
    max_force.append(test_max)
    min_force.append(test_min) 

mn = train_set.mean[1:4]
sd = train_set.stdev[1:4]
pred_list = []
metrics_list = []
model.eval()
for i,test in enumerate(test_list):
    #test_set = EncodedDataset(file_dir, encode_dir, data_sets = [test], eval_params = (train_set.mean,train_set.stdev))
    test_set = dat.ImgDataset(file_dir,file_dir,
                           data_sets=[test],
                           transform = trans_function,
                           crop_list=crop_list,
                           eval_params = (train_set.mean,train_set.stdev),
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
            pred = model(img,x)
            #pred = model(x)
        preds = np.vstack((preds,(pred.cpu().numpy()*sd)+mn))
    pred_list.append(preds)
    condition=condition_list[i]
    print(condition)
    metrics = compute_loss_metrics(preds[10:-10,:],test_loader.dataset.raw_label_array[10:-10,1:4],max_force[i],min_force[i],condition,"ENC")
    metrics_list.append(metrics)
    
import pickle    

pickle.dump(metrics_list,open("df_ENCft_test.df","wb"))
pickle.dump(pred_list,open("preds_ENCft.pred","wb"))