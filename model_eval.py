#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:05:46 2020

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

def compute_loss_metrics(predictions,labels):
    print()
    print("Summary Performance Statistics")
    print("-"*10)
    # compute MSEloss
    mean_error = np.mean(np.linalg.norm(predictions-labels,axis=1))
    print("Mean Resultant Force Error: {0:.3f}".format(mean_error))
    
    # compute normalized mean loss
    range_divisor = np.max(np.linalg.norm(labels,axis=1))-np.min(np.linalg.norm(labels,axis=1))
    normed_mean_error = np.mean(np.linalg.norm(predictions-labels,axis=1)/range_divisor,axis=0)
    print("Normalized Mean Resultant Force Error: {0:.3f}".format(normed_mean_error))
    
    # per axis errors
    per_axis_RMSE = np.sqrt(np.mean((predictions-labels)**2,axis=0))
    print("Per Axis RMSE: x:{0:.3f}, y:{1:.3f},z:{2:.3f}".format(per_axis_RMSE[0],per_axis_RMSE[1],per_axis_RMSE[2]))
    
    #normalize error
    range_divisor = np.max(labels,axis=0)-np.min(labels,axis=0)
    normed_axis_RMSE = np.sqrt(np.mean((predictions-labels)**2,axis=0))/range_divisor
    print("Per Axis normalized RMSE: x:{0:.3f}, y:{1:.3f},z:{2:.3f}".format(normed_axis_RMSE[0],normed_axis_RMSE[1],normed_axis_RMSE[2]))
    print(' ')
    
    output_dict = {'ME': mean_error,
                   'nME': normed_mean_error,
                   'Per Axis RMSE': per_axis_RMSE,
                   'Per Axis nRMSE':normed_axis_RMSE}
    
    return output_dict

def plot_trajectories(predictions,labels):

    fig, ax = plt.subplots(3,1,sharex=True)
    
    ax[0].plot(labels[:,0])
    ax[0].set_ylabel("X Force [N]")
    ax[0].plot(predictions[:,0],linewidth=1)
    ax[0].set_ylim(-5,5)
    
    ax[1].plot(labels[:,1])
    ax[1].plot(predictions[:,1],linewidth=1)
    ax[1].set_ylabel("Y Force [N]")
    ax[1].set_ylim(-5,5)
    
    ax[2].plot(labels[:,2])
    ax[2].plot(predictions[:,2],linewidth=1)
    ax[2].set_ylabel("Z Force [N]")
    ax[2].set_xlabel("Time")
    ax[2].set_ylim(-10,5)

def plot_pearson(predictions,labels):
    
    p_sorted = np.sort(predictions,axis = 0)
    l_sorted = np.sort(labels,axis=0)
    
    fig, ax = plt.subplots(1,3)
    
    ax[0].plot(l_sorted[:,0],p_sorted[:,0],'.')
    plot_unity_line(l_sorted[:,0],p_sorted[:,0],ax[0])
    ax[0].set_ylabel("X Force Pred [N]")
    ax[0].set_xlabel("X Force True [N]")
    #ax[0].set_ylim(-5,5)
    
    ax[1].plot(l_sorted[:,1],p_sorted[:,1],'.')
    plot_unity_line(l_sorted[:,1],p_sorted[:,1],ax[1])
    ax[1].set_ylabel("Y Force Pred [N]")
    ax[1].set_xlabel("Y Force True [N]")
    #ax[1].set_ylim(-5,5)
    
    ax[2].plot(l_sorted[:,2],p_sorted[:,2],'.')
    plot_unity_line(l_sorted[:,2],p_sorted[:,2],ax[2])
    ax[2].set_ylabel("Z Force Pred [N]")
    ax[2].set_xlabel("Z Force True [N]")
    #ax[2].set_ylim(-10,5)

def plot_unity_line(x,y,ax):
    
    xplot = np.linspace(np.min(x),np.max(x),num=100)
    yplot = np.linspace(np.min(x),np.max(x),num=100)
    ax.plot(xplot,yplot,color="black",linewidth=1)


def compute_GBP(model,dataloader,num_state_inputs=54,model_type="S",no_pbar=False):
    ''' 
    Computes the Guided Backpropogation values - only used for models with state information.
    right now only works for force only predictions - can be adapted for torques as well
    '''
    tqdm.write('Computing guided backprop...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        tqdm.write("using GPU acceleration")
    
    model = model.to(device,dtype=torch.float)
    model.eval()
    
    gbp_array = [np.empty((0,num_state_inputs)) for i in range(3)]
    gbp_norm = [[] for i in range(3)]
    gbp = GuidedBackprop(model)
    
    for n in range(3):
        with tqdm(total = len(dataloader),disable=no_pbar) as pbar:
            for img,state,labels in dataloader:
                state = state.to(device,dtype=torch.float)
                if model_type!="S":
                    img=img.to(device,dtype=torch.float)
                    attributions = gbp.attribute((img,state),target=n)
                    gbp_array[n] = np.vstack((gbp_array[n],attributions[1].cpu().numpy()))
                else:
                    attributions = gbp.attribute(state,target=n)
                    gbp_array[n] = np.vstack((gbp_array[n],attributions.cpu().numpy()))
                pbar.update(1)
    
        gbp_norm[n] = gbp_array[n]-np.min(gbp_array[n],axis=0)
        gbp_norm[n] = gbp_norm[n]/np.reshape(np.sum(gbp_norm[n],axis=1),(-1,1))
    
    return gbp_norm

def compute_and_plot_gbp(gbp_data,feature_list,aggregate_force=True,suppress_plots=False):
    
    # convert calculate mean gbp value of each feature
    gbp_mean = np.mean(np.array(gbp_data),axis=1)
    df = pd.DataFrame(gbp_mean, index=['x','y','z'], columns=feature_list)
    df_long = pd.melt(df,value_name = 'gbp',var_name = 'feature',ignore_index=False).reset_index(inplace=False) 
    df_long = df_long.rename(columns={'index':'axis'})
    
    if not suppress_plots:
        if aggregate_force:
            sns.catplot(data=df_long,x='feature',y='gbp',kind='bar',ci='sd')
        else:
            sns.catplot(data=df_long,x='feature',y='gbp',hue='axis',kind='bar')
        f = plt.gcf()
        f.tight_layout()
    
    return df_long
    
def plot_heatmaps(gbp_data,predictions,labels,feature_list):
    
    directions = ["X","Y","Z"]
    
    for n in range(3):   
        fig = plt.figure()
        fig.suptitle('Direction '+ directions[n], fontsize=10)
        ax = [[],[]]
        ax[0] = plt.subplot2grid((6, 1), (0, 0))
        ax[1] = plt.subplot2grid((6, 1), (1, 0),rowspan=5,sharex = ax[0])

        ax[0].plot(labels[:,n])
        ax[0].set_ylabel("Force [N]")
        ax[0].plot(predictions[:,n])
        ax[0].set_ylim(-2.5,2.5)
        
        ax[1].imshow(gbp_data[n].transpose(), cmap='jet', interpolation='nearest',aspect="auto")
        ax[1] = plt.gca()
        ax[1].set_yticks(np.arange(len(feature_list)))
        ax[1].set_yticklabels(feature_list)
        
        fig.tight_layout()
        
        
'global variable'
qty = ['px','py','pz','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz',
       'q1','q2','q3','q4','q5','q6','q7',
       'vq1','vq2','vq3','vq4','vq5','vq6','vq7',
       'tq1','tq2','tq3','tq4','tq5','tq6','tq7',
       'q1d','q2d','q3d','q4d','q5d','q6d','q7d',
       'tq1d','tq2d','tq3d','tq4d','tq5d','tq6d','tq7d',
       'psm_fx','psm_fy','psm_fz','psm_tx','psm_ty','psm_tz']


if __name__ == "__main__":

    model_type = "VS"
    feat_extract = True
    
    crop_list = []
    '''
    for i in range(1,14):
        crop_list.append((57,320,462,462))
    for i in range(14,17):
        crop_list.append((40,400,462,462))
    for i in range(17,20):
        crop_list.append((57,180,462,462))
    for i in range(20,23):
        crop_list.append((70,145,462,462))
    crop_list.append((30,250,462,462))
    '''
    for i in range(1,8):
        crop_list.append((50,350,300,300))
        
    # Define a transformation for the images
    trans_function = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    # load the model
    if model_type == "VS":
        model = mdl.StateVisionModel(30, 54, 3,feature_extract=feat_extract)
    elif model_type == "S":
        model  = mdl.StateModel(54, 3)
    
    if model_type!="S" and feat_extract:
        model_weights = torch.load("best_modelweights_"+model_type+"_ft.dat")
    else:
        model_weights = torch.load("best_modelweights_"+model_type+".dat")
    model.load_state_dict(model_weights)
    
    # load the dataset
    file_dir = '../ML dvrk 081320' # define the file directory for dataset
    train_list = [1,2,3,5,6]
    test_list = [4,7]
    config_dict={'file_dir':file_dir,
                 'include_torque': False,
                 'custom_state': None,
                 'batch_size': 16,
                 'crop_list': crop_list,
                 'trans_function': trans_function}
    
    loader_dict,loader_sizes = dat.init_dataset(train_list,test_list,test_list,model_type,config_dict)
    test_loader = loader_dict['test']
    
    plt.close('all')
    # compute the loss and other performance metrics
    predictions = mdl.evaluate_model(model,test_loader,model_type = model_type)
    compute_loss_metrics(predictions,test_loader.dataset.label_array[:,1:4])
    plot_trajectories(predictions,test_loader.dataset.label_array[:,1:4])
    plot_pearson(predictions,test_loader.dataset.label_array[:,1:4])
    
    gbp_data = compute_GBP(model,test_loader,model_type=model_type)
    plot_heatmaps(gbp_data,predictions,test_loader.dataset.label_array[:,1:4],qty)
    df_gbp_means = compute_and_plot_gbp(gbp_data,qty,True)
    df_gbp_means.groupby('feature').mean().sort_values(by='gbp',ascending=False)
