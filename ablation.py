#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study Script

Created on Wed Sep  9 16:48:18 2020
@author: charm
"""
import models as mdl
import model_eval
import dataset_lib as dat
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as transforms
import seaborn as sns
import pandas as pd
import pickle

def plot_ablations_metrics(quantity,perf_metrics,param_counts,removed_features):
    
    df = pd.DataFrame(perf_metrics)
    df['removed_features'] = removed_features
    df['param_counts'] = param_counts
    df[['RMSEx','RMSEy','RMSEz']] = pd.DataFrame(df['Per Axis RMSE'].values.tolist(),index=df.index)
    df[['nRMSEx','nRMSEy','nRMSEz']] = pd.DataFrame(df['Per Axis nRMSE'].values.tolist(),index=df.index)
    df = df.drop(labels=['Per Axis RMSE','Per Axis nRMSE'],axis=1)
    df_melt = pd.melt(frame=df,
                      id_vars = ['removed_features','param_counts'],
                      value_vars= quantity ,
                      value_name='Error Value',
                      var_name='Type of Error')
    sns.catplot(data=df_melt,x='removed_features',y = 'Error Value',hue='Type of Error',kind='point')
    
    return df_melt

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_ablations(model,num_ablations):
    
    '''set up some persistent tracking variables'''
    remove_features = [] # list of features we are removing
    metrics_list = [] # list storing dictionary of performance metrics
    # feature indexes
    full_state_index = np.arange(7,61)
    input_state = 54
    # create loss function
    criterion = nn.MSELoss(reduction='sum')
    # define optimization method
    optimizer = opt.Adam(model.parameters(),lr=0.01)
    param_count = []
    param_count.append(count_params(model))
    current_feature_list = np.array(qty)
    
    # create the dataloader
    dataloaders,dataset_sizes = dat.init_dataset(train_list,val_list,val_list,model_type,config_dict)
    
    print('evaluating full model predictions...')
    predictions = mdl.evaluate_model(model, dataloaders['test'], model_type=model_type,no_pbar=True)
    # compute the loss statistics
    print('computing full model performance metrics...')
    metrics = model_eval.compute_loss_metrics(predictions, dataloaders['test'].dataset.label_array[:,1:4])
    metrics_list.append(metrics)
    print('Performance Summary of Full Model:')
    print(metrics)
    
    print('Running ablation study on model type:' + model_type)
    
    for iteration in range(num_ablations):
        print('-'*10)
        print('Begin ablation run: {}/{}'.format(iteration+1,num_ablations))
        print('-'*10)

        # compute the backprop values:
        gbp_data = model_eval.compute_GBP(model,dataloaders['test'],
                                          num_state_inputs=input_state,
                                          model_type=model_type,
                                          no_pbar=True)
        # evaluate means
        df_gbp_means = model_eval.compute_and_plot_gbp(gbp_data,current_feature_list,True,suppress_plots=True)
        # group by feature type and rank by value
        df_gbp_means = df_gbp_means.groupby('feature').mean().sort_values(by='gbp',ascending=False).reset_index()
        # get top ranking value and append to removal list
        feature_to_remove = df_gbp_means.iloc[0,0]
        print("removing " + feature_to_remove + "...")
        remove_features.append(feature_to_remove)
        # create the mask
        mask = np.isin(qty,remove_features,invert=True)
        # mask the full state vector in config_dict global variable
        config_dict['custom_state'] = full_state_index[mask]
        current_feature_list = np.array(qty)[mask] #update the current feature list
        # decrease the input dimension of the model by one
        input_state = input_state - 1
        
        # redefine the models
        print('redefining model with input state dims: {}'.format(input_state))
        if model_type == "VS":
            model = mdl.StateVisionModel(30, input_state, 3,feature_extract=feat_extract)
        elif model_type == "S":
            model  = mdl.StateModel(input_state, 3)
    
        # recalculate the number of parameters
        param_count.append(count_params(model))
        
        # redefine the optimizer
        optimizer = opt.Adam(model.parameters(),lr=0.01)
        
        # redefine the dataloader
        dataloaders,dataset_sizes = dat.init_dataset(train_list,val_list,val_list,model_type,config_dict)
        
        # retrain the model
        model,train_history,val_history = mdl.train_model(model,
                                                          criterion, optimizer,
                                                          dataloaders, dataset_sizes,
                                                          num_epochs=50,
                                                          model_type= model_type,
                                                          weight_file=weight_file,
                                                          no_pbar=True)
        print('retraining completed')
        # do inference
        print('evaluating model predictions...')
        predictions = mdl.evaluate_model(model, dataloaders['test'], model_type=model_type,no_pbar=True)
        # compute the loss statistics
        print('computing performance metrics...')
        metrics = model_eval.compute_loss_metrics(predictions, dataloaders['test'].dataset.label_array[:,1:4])
        metrics_list.append(metrics)
        print('Performance Summary:')
        print(metrics)
    
    return remove_features,param_count,metrics_list


'''global variables'''

qty = ['px','py','pz','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz',
       'q1','q2','q3','q4','q5','q6','q7',
       'vq1','vq2','vq3','vq4','vq5','vq6','vq7',
       'tq1','tq2','tq3','tq4','tq5','tq6','tq7',
       'q1d','q2d','q3d','q4d','q5d','q6d','q7d',
       'tq1d','tq2d','tq3d','tq4d','tq5d','tq6d','tq7d',
       'psm_fx','psm_fy','psm_fz','psm_tx','psm_ty','psm_tz']

crop_list = []
for i in range(1,8):
    crop_list.append((50,350,300,300))
    
file_dir = '../ML dvrk 081320'

# Define a transformation for the images
trans_function = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
force_align=True
train_list = [1,2,3,5,6]
val_list = [4,7]
config_dict={'file_dir':file_dir,
             'include_torque': False,
             'custom_state': None,
             'batch_size': 16,
             'crop_list': crop_list,
             'spatial_forces': force_align,
             'trans_function': trans_function}

model_type = "S"
feat_extract = True

weight_file = "best_modelweights_ablate_temp.dat"

if __name__ == "__main__":

    # load the model
    if model_type == "VS":
        model = mdl.StateVisionModel(30, 54, 3,feature_extract=feat_extract)
    elif model_type == "S":
        model  = mdl.StateModel(54, 3)
    
    weight_file =  weight_file = "best_modelweights_" + model_type
    
    if model_type!="S" and feat_extract:
        weight_file="best_modelweights_" + model_type + "_ft"
        
    if force_align and model_type!= "V" :
        weight_file = weight_file + "_faligned"
        
    weight_file = weight_file + ".dat"
   
    model.load_state_dict(torch.load(weight_file))

    removed_features,param_counts,perf_metrics =run_ablations(model,num_ablations=30)
    
    #metrics = ['ME','RMSEx','RMSEy','RMSEz'] 
    #metrics = ['nRMSEx','nRMSEy','nRMSEz']
    
    metrics = ['ME','RMSEx','RMSEy','RMSEz'] + ['nRMSEx','nRMSEy','nRMSEz']
    result_frame = plot_ablations_metrics(metrics, perf_metrics, param_counts, ['orig']+removed_features)
    
    save_file = open('091120_ablation_faligned.df','wb')
    pickle.dump(result_frame,save_file)
    save_file.close()
    #stuff to do:
    # perform a few runs and see if the features are stable.
    # do some averaging
    # is it better to remove in a more "logical way?"
    
    
        
        