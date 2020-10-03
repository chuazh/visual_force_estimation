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

def compute_loss_metrics(predictions,labels,condition,model_type):
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
        plt.xticks(rotation=70)
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
        
def summarize_metrics(metrics_list,metric,labels):
    
    # create the dataframe
    
    for metric_dict,condition in zip(metrics_list,labels):
        metric_dict['condition'] = condition
    
    df = pd.DataFrame(metrics_list)
    df['Per Axis RMSE'].apply(pd.Series)
    df['Per Axis nRMSE'].apply(pd.Series)
    
    sns.catplot(x='condition',y=metric,data=df,kind='bar')
    
    return df
 
       
'global variable'
qty = ['px','py','pz','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz',
       'q1','q2','q3','q4','q5','q6','q7',
       'vq1','vq2','vq3','vq4','vq5','vq6','vq7',
       'tq1','tq2','tq3','tq4','tq5','tq6','tq7',
       'q1d','q2d','q3d','q4d','q5d','q6d','q7d',
       'tq1d','tq2d','tq3d','tq4d','tq5d','tq6d','tq7d',
       'psm_fx','psm_fy','psm_fz','psm_tx','psm_ty','psm_tz']


if __name__ == "__main__":

    model_type = "VS_deep"
    feat_extract = False
    force_align = False
    
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
    for i in range(1,37):
        #crop_list.append((50,350,300,300))
        crop_list.append((270-150,480-150,300,300))
        
    # Define a transformation for the images
    trans_function = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    # load the model
    if model_type == "VS":
        model = mdl.StateVisionModel(30, 54, 3,feature_extract=feat_extract)
    elif model_type == "S":
        model  = mdl.StateModel(54, 3)
    elif model_type == "VS_deep":
        model = mdl.StateVisionModel_deep(30, 54, 3)
    
    weight_file =  weight_file = "best_modelweights_" + model_type
    
    if model_type!="S" and feat_extract:
        weight_file="best_modelweights_" + model_type + "_ft"
        
    if force_align and model_type!= "V" :
        weight_file = weight_file + "_faligned"
        
    weight_file = weight_file + ".dat"
    
    model.load_state_dict(torch.load(weight_file))
    
    # load the dataset
    file_dir = '../experiment_data' # define the file directory for dataset
    train_list = [1]
    val_list = [1]
    config_dict={'file_dir':file_dir,
             'include_torque': False,
             'spatial_forces': force_align,
             'custom_state': None,
             'batch_size': 32,
             'crop_list': crop_list,
             'trans_function': trans_function}
    
    
    
    #%%   Manual Plotting
    
    test_list = [34]  
    loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)
    test_loader = loader_dict['test']

    #plt.close('all')
    predictions = mdl.evaluate_model(model,test_loader,model_type = model_type)
    # compute the loss and other performance metrics
    metrics = compute_loss_metrics(predictions,test_loader.dataset.label_array[:,1:4],'new_material',"VS")
    
    plot_trajectories(predictions,test_loader.dataset.label_array[:,1:4])
    plot_pearson(predictions,test_loader.dataset.label_array[:,1:4])
    gbp_data = compute_GBP(model,test_loader,model_type=model_type)
    plot_heatmaps(gbp_data,predictions,test_loader.dataset.label_array[:,1:4],qty)
    df_gbp_means = compute_and_plot_gbp(gbp_data,qty,False)
    df_gbp_means.groupby('feature').mean().sort_values(by='gbp',ascending=False)
    
    #%%  
    
    predictions_list = []
    metrics_list = []
    #test_list_full = [2,6,9,13,16,20]
    #condition_list = ['center','center','right','right','left','left']
    
    test_list_full =  [4,11,18,22,23,24,25,26,27,28,29,32,33,34,36]
    condition_list = ['center','right','left',
                      'right_less','right_less',
                      'right_more','right_more',
                      'left_less','left_less',
                      'left_more','left_more',
                      'new_tool','new_tool',
                      'new_material','new_material']
    
    for test,condition in zip(test_list_full,condition_list):
        test_list = [test]
        loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)
        test_loader = loader_dict['test']
        #plt.close('all')
        predictions = mdl.evaluate_model(model,test_loader,model_type = model_type)
        predictions_list.append(predictions)
        
    # compute the loss and other performance metrics
        metrics = compute_loss_metrics(predictions[10:-10,:],test_loader.dataset.label_array[10:-10,1:4],condition,model_type)
        metrics_list.append(metrics)
    
    # create dataframe
    df_metrics = pd.DataFrame(metrics_list)
    
    import pickle
    df_filedir = 'df_'+model_type+'_test.df'
    pickle.dump(df_metrics,open(df_filedir,'wb'))

    #%% Visualization
    
    df_S = pickle.load(open('df_S_test.df','rb'))
    #df_S = df_metrics
    df_deep = pickle.load(open('df_VS_test.df','rb'))
    df_VS = pickle.load(open('df_VS_test.df','rb'))
    dyn_model_data = pickle.load(open('../dvrk_dynamic_model/dvrk_dynamics_identification/dynamic_model_preds.dat','rb'), encoding='latin1')
    df_dyn = pd.DataFrame(dyn_model_data['metric_data'])
    
    
    df_merge = pd.concat([df_S,df_dyn,df_VS])
    #df_merge = pd.concat([df_S,df_VS])
    df_merge = pd.melt(df_merge,id_vars=['condition','model'],value_vars=['Per Axis nRMSEx','Per Axis nRMSEy','Per Axis nRMSEz'],var_name = 'metric',value_name='value')
    sns.catplot(data=df_merge,x='model',y='value',col='condition',kind='bar')
    #sns.catplot(x='model',y='value',col='condition',col_wrap=3,hue='metric',col_order=['right_less','right','right_more','left_less','left','left_more',],data=df_merge.loc[(df_merge['condition']!='center') & (df_merge['condition']!='new_tool')],kind='bar')
    #sns.catplot(x='model',y='value',col='condition',hue='metric',col_wrap=3,data=df_merge.loc[(df_merge['condition']=='center') | (df_merge['condition']=='new_tool')|(df_merge['condition']=='new_material')],kind='bar')
    df_merge.groupby(['metric','condition','model']).agg({'value':'mean'})

    #%%
    
    for i in range(3):
        for k in range(3):
            ax[i,k].plot(reference_force[10:-10,i])
        ax[i,0].plot(predictions_VS[10:-10,i],linewidth=0.75)
        ax[i,2].plot(predictions_S[10:-10,i],linewidth=0.75)
        ax[i,1].plot(predictions_D[:,i],linewidth=0.75)