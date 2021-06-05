#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:25:56 2021

@author: charm
"""


import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
import seaborn as sns
import pickle

import numpy as np

def plot_trajectories(index,preds,lookup,experiment_path,ax,plot_gt = True,thresh_mask=None):
    
    predictions = preds[index]
    exp_num = lookup[index]
    labels = np.loadtxt(experiment_path+'/labels_'+str(exp_num)+'.txt',delimiter=',')
    
    if thresh_mask is not None:
        print('computing RMSE, masking z forces less than '+str(thresh_mask))
        thresh_mask = np.where(labels[:,3]>thresh_mask)[0]
        print(np.sqrt(np.mean(np.power(predictions[thresh_mask,:]-labels[thresh_mask,1:4],2),0)))
    else:
        print(np.sqrt(np.mean(np.power(predictions-labels[:,1:4],2),0)))
    
    time = labels[:,0]
    if plot_gt:
        ax[0].plot(time,labels[:,1],color="black")
        ax[1].plot(time,labels[:,2],color="black")
        ax[2].plot(time,labels[:,3],color="black")
    
    ax[0].set_ylabel("X Force [N]")
    ax[0].plot(time,predictions[:,0],linewidth=1)
    ax[0].set_ylim(-5,5)
    
    ax[1].plot(time,predictions[:,1],linewidth=1)
    ax[1].set_ylabel("Y Force [N]")
    ax[1].set_ylim(-5,5)
    
    ax[2].plot(time,predictions[:,2],linewidth=1)
    ax[2].set_ylabel("Z Force [N]")
    ax[2].set_xlabel("Time")
    ax[2].set_ylim(-10,5)
    
    return ax

#%%

df_S = pickle.load(open('processed_data/df_S_test.df','rb'))
df_VS = pickle.load(open('processed_data/df_VS_test.df','rb'))
df_V = pickle.load(open('processed_data/df_V_test.df','rb'))
df_TFN = pickle.load(open('df_VS_test_TFN.df','rb'))
df_ENC = pd.DataFrame(pickle.load(open('df_VAE_test.df','rb')))

test_numbering_list = [1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,2,2,2,1,1,1]
df_S['test_number'] = test_numbering_list
df_VS['test_number'] = test_numbering_list
df_V['test_number'] = test_numbering_list
df_TFN['test_number'] = test_numbering_list
df_ENC['test_number'] = test_numbering_list
df_TFN['model'] = "TFN"
df_ENC['model'] = "VAE"
df_merge = pd.concat([df_S,df_V,df_VS,df_TFN,df_ENC])
df_RMSE = pd.melt(df_merge,id_vars=['condition','model'],value_vars=['Per Axis RMSEx','Per Axis RMSEy','Per Axis RMSEz'],var_name = 'metric',value_name='value')
df_nRMSE = pd.melt(df_merge,id_vars=['condition','model'],value_vars=['Per Axis nRMSEx','Per Axis nRMSEy','Per Axis nRMSEz'],var_name = 'metric',value_name='value')

sns.catplot(x='condition',
            y='value',
            hue="model",
            row='metric',
            order=['left_more','left','left_less','center','right_less','right','right_more','z_low','z_mid','z_high'],
            data=df_RMSE,
            kind='point',
            ci=None)
sns.catplot(x='condition',
            y='value',
            hue="model",
            row='metric',
            order=['new_material','center','new_tool'],
            data=df_RMSE,
            kind='point',
            ci=None)

preds_S = pickle.load(open('processed_data/preds_S.preds','rb'))
preds_V = pickle.load(open('processed_data/preds_V.preds','rb'))
preds_VS = pickle.load(open('processed_data/preds_VS.preds','rb'))
preds_TFN = pickle.load(open('preds_VS_TFN.preds','rb'))
preds_ENC = pickle.load(open('preds_ENC.preds','rb'))
experiment_data_path = "../../experiment_data"

test_list_full =  [4,11,18,
                   22,23,
                   24,25,
                   26,27,
                   28,29,
                   32,33,
                   34,36,
                   37,38,39,
                   45,46,47]


#%%
index = 18
fig, ax = plt.subplots(3,1,sharex=True)
ax = plot_trajectories(index, preds_S, test_list_full, experiment_data_path,ax,True,-5)
ax = plot_trajectories(index, preds_V, test_list_full, experiment_data_path,ax,False,-5)
ax = plot_trajectories(index, preds_VS, test_list_full, experiment_data_path,ax,False,-5)
ax = plot_trajectories(index, preds_TFN, test_list_full, experiment_data_path,ax,False,-5)
fig.suptitle('z_mid')

index = 19
fig, ax = plt.subplots(3,1,sharex=True)
ax = plot_trajectories(index, preds_S, test_list_full, experiment_data_path,ax,True,-5)
ax = plot_trajectories(index, preds_V, test_list_full, experiment_data_path,ax,False,-5)
ax = plot_trajectories(index, preds_VS, test_list_full, experiment_data_path,ax,False,-5)
ax = plot_trajectories(index, preds_TFN, test_list_full, experiment_data_path,ax,False,-5)
fig.suptitle('z_high')

index = 20
fig, ax = plt.subplots(3,1,sharex=True)
ax = plot_trajectories(index, preds_S, test_list_full, experiment_data_path,ax,True,-5)
ax = plot_trajectories(index, preds_V, test_list_full, experiment_data_path,ax,False,-5)
ax = plot_trajectories(index, preds_VS, test_list_full, experiment_data_path,ax,False,-5)
ax = plot_trajectories(index, preds_TFN, test_list_full, experiment_data_path,ax,False,-5)
fig.suptitle('z_low')


index = 12
fig, ax = plt.subplots(3,1,sharex=True)
ax = plot_trajectories(index, preds_S, test_list_full, experiment_data_path,ax)
ax = plot_trajectories(index, preds_V, test_list_full, experiment_data_path,ax,False)
ax = plot_trajectories(index, preds_VS, test_list_full, experiment_data_path,ax,False)
ax = plot_trajectories(index, preds_TFN, test_list_full, experiment_data_path,ax,False,-5)

#%% ALL the DATA

df_S = pickle.load(open('processed_data_nRMSE_new/df_S_test.df','rb'))
df_VS = pickle.load(open('processed_data_nRMSE_new/df_VS_test.df','rb'))
df_V = pickle.load(open('processed_data_nRMSE_new/df_V_test.df','rb'))
df_VS_F = pickle.load(open('processed_data_nRMSE_new/df_VS_ablate_F_test.df','rb'))
df_VS_P = pickle.load(open('processed_data_nRMSE_new/df_VS_ablate_P_test.df','rb'))
df_S_F = pickle.load(open('processed_data_nRMSE_new/df_S_ablate_F_test.df','rb'))
df_S_P = pickle.load(open('processed_data_nRMSE_new/df_S_ablate_P_test.df','rb'))

dyn_model_data = pickle.load(open('processed_data_nRMSE_new/dynamic_model_preds_test_031421.dat','rb'),encoding='latin1')
df_dyn = pd.DataFrame(dyn_model_data['metric_data'])

df_RNN = pickle.load(open('processed_data_nRMSE_new/df_V_RNN_test.df','rb'))

test_numbering_list = [1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,2,2,2,1,1,1]
df_S['test_number'] = test_numbering_list
df_VS['test_number'] = test_numbering_list
df_V['test_number'] = test_numbering_list
df_VS_P['test_number'] = test_numbering_list
df_VS_F['test_number'] = test_numbering_list
df_S_P['test_number'] = test_numbering_list
df_S_F['test_number'] = test_numbering_list
df_dyn['test_number'] = test_numbering_list
df_RNN['test_number'] = test_numbering_list

df_VS_F['model'] = 'VS_pos_only'
df_VS_P['model'] = 'VS_force_only'
df_S_F['model'] = 'S_pos_only'
df_S_P['model'] = 'S_force_only'

df_merge = pd.concat([df_S,df_V,df_VS,df_VS_P,df_VS_F,df_S_P,df_S_F,df_dyn,df_RNN])
df_merge = pd.melt(df_merge,id_vars=['condition','model','test_number'],value_vars=['Per Axis RMSEx','Per Axis RMSEy','Per Axis RMSEz'],var_name = 'metric',value_name='value')
df_merge.groupby(['metric','condition','model']).agg({'value':'mean'})
#df_merge.to_csv("nRMSE_check_031721.csv")
