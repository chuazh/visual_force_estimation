#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:08:57 2021

@author: charm
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd


def compute_RMSE(pred,label,model,miu,sigma,threshold=None):
    
    if model == "D":
        pred = pred[70:,:]
    elif model == "RNN":
        # the RNN batch size was 250 thus the padlength was:
        pad_length =  pred.shape[0] - (label.shape[0]-80)
        pred = pred[:-pad_length-10,:3]
    else:
        pred = pred[80:-10,:]
    
    if model != "D":    
        pred = (pred*sigma)+miu #NOTE THAT THIS ASSUMES THAT THE PREDS AREN'T NORMALIZED
    
    if threshold is not None:
        label = label[80:-10,1:4]
        threshold_idx = np.where(label[:,2]>threshold)[0]
        per_axis_RMSE = np.sqrt(np.mean((pred[threshold_idx,:]-label[threshold_idx,:])**2,axis=0))
    else:
        # compute the per axis RMSE
        per_axis_RMSE = np.sqrt(np.mean((pred-label[80:-10,1:4])**2,axis=0))
    print("Per Axis RMSE: x:{0:.3f}, y:{1:.3f},z:{2:.3f}".format(per_axis_RMSE[0],per_axis_RMSE[1],per_axis_RMSE[2]))
    
    return per_axis_RMSE


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

test_numbering_list = [1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,2,2,2,1,1,1]
#%% DATA LOAD TEST
pred_dict = {}
pred_dict['V'] = pickle.load(open("preds_V.preds","rb"))
pred_dict['VS'] = pickle.load(open("preds_VS.preds","rb"))
pred_dict['S'] = pickle.load(open("preds_S.preds","rb"))
pred_dict['RNN'] = pickle.load(open("../preds_V_RNN.preds","rb"))
pred_dict['D'] = pickle.load(open('dynamic_model_preds_test_031421.dat','rb'), encoding='latin1')['pred_data']
#pred_dict['D'] = pickle.load(open('../../dynamic_model_preds_test_031421.dat','rb'), encoding='latin1')['pred_data']
pred_dict['VS-F-only'] =  pickle.load(open("preds_VS_ablate_P.preds","rb"))
pred_dict['VS-P-only'] =  pickle.load(open("preds_VS_ablate_F.preds","rb"))
pred_dict['S-F-only'] =  pickle.load(open("preds_S_ablate_P.preds","rb"))
pred_dict['S-P-only'] =  pickle.load(open("preds_S_ablate_F.preds","rb"))
pred_dict['VS-TFN'] = pickle.load(open("preds_VS_TFN.preds","rb"))
pred_dict['V-small'] = pickle.load(open("preds_V_shallow.preds","rb"))
pred_dict['VS-VAE128'] = pickle.load(open("../preds_VAE.pred","rb"))
pred_dict['VS-VAE56'] = pickle.load(open("preds_VAE56.pred","rb"))

#model_list = ['V','VS','S','RNN','D','VS-F-only','VS-P-only','S-F-only','S-P-only','VS-TFN','V-small']
model_list = ['V','VS','S','VS-VAE128','VS-TFN']

test_RMSE = pd.DataFrame(columns=['model','condition','RMSEx','RMSEy','RMSEz','test_number'])

for test in test_list_full:
    tr = test_list_full.index(test)
    label = np.loadtxt("../../../experiment_data/labels_"+str(test)+".txt",delimiter=",")
    print('condition: ' + condition_list[tr])
    for model in model_list:
        print('model: ' + model)
        #rmse = compute_RMSE(pred_dict[model][tr], label, model,np.zeros((3,)),np.ones((3,)))
        rmse = compute_RMSE(pred_dict[model][tr], label, model,np.zeros((3,)),np.ones((3,)),-5) #hard code zeros to miu and sigma because I already normalizd in model_eval.py
        df_dict = {'model':model,'condition':condition_list[tr],'RMSEx':rmse[0],'RMSEy':rmse[1],'RMSEz':rmse[2],'test_number':test_numbering_list[tr]}
        test_RMSE=test_RMSE.append(pd.DataFrame(df_dict,index=[0]),ignore_index=True)
        #train_RMSE=train_RMSE.append(pd.DataFrame(df_dict,index=[0]),ignore_index=True)
        #val_RMSE=val_RMSE.append(pd.DataFrame(df_dict,index=[0]),ignore_index=True)

test_RMSE['mean'] = test_RMSE[['RMSEx','RMSEy','RMSEz']].mean(axis=1)
summary_test_RMSE=test_RMSE[['model','condition','mean']].groupby(['condition','model']).mean()
#test_RMSE.to_csv("RMSE_all_thresh5.csv")
#%% Check
import seaborn as sns

df_long= pd.melt(test_RMSE,id_vars=['condition','model','test_number'],value_vars=['RMSEx','RMSEy','RMSEz'],var_name = 'metric',value_name='value')
sns.catplot(x='condition',y='value',hue="model",row='metric',order=['right_more','right','right_less','center','left_less','left','left_more','z_low','z_mid','z_high'],data=df_long,kind='point',ci=None)
sns.catplot(x='condition',y='value',hue="model",row='metric',order=['new_tool','center','new_material'],data=df_long,kind='point',ci=None)

sns.catplot(x='condition',y='mean',hue="model",order=['right_more','right','right_less','center','left_less','left','left_more','z_low','z_mid','z_high','new_material','new_tool'],data=test_RMSE,kind='point',ci=None)
#%% TRAJ PLOT

def plot_trajectory_from_preds(pred,label,model,miu,sigma,ax,plot_label=False,plot_range=None):
    
    # we need to cut out 10 steps from the back (due to dynamic)
    # we need to cut out 60 steps from the front  (due to RNN)
    
    if model == "D":
        pred = pred[70:,:]
    elif model == "RNN":
        # the RNN batch size was 250 thus the padlength was:
        pad_length =  pred.shape[0] - (label.shape[0]-80)
        pred = pred[:-pad_length-10,:3]
    else:
        pred = pred[80:-10,:]
    
    label = label[80:-10,:]
    
    if model != "D":
        pred = (pred*sigma)+miu #NOTE THAT THIS ASSUMES THAT THE PREDS AREN'T NORMALIZED
    
    # plot each of the axes:
    for i in range(3):
        if plot_range != None:
            start_idx = plot_range[0]
            end_idx = plot_range[1]
            if plot_label:
                ax[i].plot(label[start_idx:end_idx,0],label[start_idx:end_idx,i+1],linewidth=1,color="black",label="GT")
            ax[i].plot(label[start_idx:end_idx,0],pred[start_idx:end_idx,i],linewidth=0.75,label=model)
        else:
            if plot_label:
                ax[i].plot(label[:,0],label[:,i+1],linewidth=1,color="black",label="GT")
            ax[i].plot(label[:,0],pred[:,i],linewidth=0.75,label=model)
    
    output = np.concatenate((np.expand_dims(label[:,0],axis=1),pred[:,:3]),axis=1)
    
    return output

#already normalized the outputs...
miu = np.zeros((3,))
sigma = np.ones((3,))

tr_num=33
tr = test_list_full.index(tr_num)
condition = condition_list[tr]
fig,ax = plt.subplots(nrows=3,ncols=1,sharex=True)
labels = np.loadtxt("../../../experiment_data/labels_"+str(tr_num)+".txt",delimiter=",")
output_labels = labels[80:-10,:4]

plot_length=20
max_idx = labels.shape[0]-90-(plot_length*30)
start_idx = np.random.randint(0,max_idx)
plot_range = (start_idx,start_idx+(plot_length*30))
              
output_V = plot_trajectory_from_preds(pred_dict['V'][tr],labels,"V",miu,sigma,ax,True,plot_range=plot_range)
output_VS = plot_trajectory_from_preds(pred_dict['VS'][tr],labels,"VS",miu,sigma,ax,plot_range=plot_range)
#output_S = plot_trajectory_from_preds(pred_dict['S-F-only'][tr],labels,"S",miu,sigma,ax,plot_range=plot_range)
#output_TFN = plot_trajectory_from_preds(pred_dict['VS-TFN'][tr],labels,"TFN",miu,sigma,ax,plot_range=plot_range)
output_VAE = plot_trajectory_from_preds(pred_dict['VS-VAE'][tr],labels,"Vshal",miu,sigma,ax,False,plot_range=plot_range)
#output_RNN = plot_trajectory_from_preds(pred_dict['RNN'][tr],labels,"RNN",miu,sigma,ax,plot_range=plot_range)
#output_D = plot_trajectory_from_preds(pred_dict['D'][tr],labels,"D",miu,sigma,ax,plot_range=plot_range)

handles,labels=ax[2].get_legend_handles_labels()
plt.legend(handles,labels,ncol=6,loc="lower center",bbox_to_anchor=(0.5,-0.65))

# combine all the array:
output_frame = np.hstack((output_labels,output_V[:,1:],output_VS[:,1:],output_S[:,1:],output_RNN[:,1:],output_D[:,1:]))
col =['time','GT_x','GT_y','GT_z','V_x','V_y','V_z','VS_x','VS_y','VS_z','S_x','S_y','S_z','RNN_x','RNN_y','RNN_z','D_x','D_y','D_z']
# make it into a data frame
df_traj = pd.DataFrame(output_frame,columns=col)
df_traj['condition'] = condition

#%%
df_traj.to_csv("center_traj.csv")