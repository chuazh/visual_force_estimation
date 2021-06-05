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

import pickle

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

def plot_trajectories(predictions,labels):

    fig, ax = plt.subplots(3,1,sharex=True)
    
    time = np.arange(0,predictions.shape[0]/30,1/30)
    
    ax[0].plot(time,labels[:,0])
    ax[0].set_ylabel("X Force [N]")
    ax[0].plot(time,predictions[:,0],linewidth=1)
    ax[0].set_ylim(-5,5)
    
    ax[1].plot(time,labels[:,1])
    ax[1].plot(time,predictions[:,1],linewidth=1)
    ax[1].set_ylabel("Y Force [N]")
    ax[1].set_ylim(-5,5)
    
    ax[2].plot(time,labels[:,2])
    ax[2].plot(time,predictions[:,2],linewidth=1)
    ax[2].set_ylabel("Z Force [N]")
    ax[2].set_xlabel("Time")
    ax[2].set_ylim(-10,5)

def plot_pearson_grid(axis_array,position,plot_color,predictions,labels):
    
    #p_sorted = np.take_along_axis(predictions,np.argsort(labels,axis=0),axis=0)
    #l_sorted = np.sort(labels,axis=0)
    p_sorted = predictions
    l_sorted = labels
    j = position
    
    for i in range(3):
        axis_array[i,j].plot(l_sorted[:,i],p_sorted[:,i],'.',color=plot_color,markersize=0.5)
        
    return axis_array

def plot_pearson(predictions,labels):
    
    p_sorted = np.take_along_axis(predictions,np.argsort(labels,axis=0),axis=0)
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
    return ax


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

qty_full = ['t','fx','fy','fz','tx','ty','tz',
       'px','py','pz','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz',
       'q1','q2','q3','q4','q5','q6','q7',
       'vq1','vq2','vq3','vq4','vq5','vq6','vq7',
       'tq1','tq2','tq3','tq4','tq5','tq6','tq7',
       'q1d','q2d','q3d','q4d','q5d','q6d','q7d',
       'tq1d','tq2d','tq3d','tq4d','tq5d','tq6d','tq7d',
       'psm_fx','psm_fy','psm_fz','psm_tx','psm_ty','psm_tz',
       'J1','J2','J3','J4','J5','J6','J1','J2','J3','J4','J5','J6',
       'J1','J2','J3','J4','J5','J6','J1','J2','J3','J4','J5','J6',
       'J1','J2','J3','J4','J5','J6','J1','J2','J3','J4','J5','J6']

force_features = ['tq1','tq2','tq3','tq4','tq5','tq6','tq7',
   'q1d','q2d','q3d','q4d','q5d','q6d','q7d',
   'tq1d','tq2d','tq3d','tq4d','tq5d','tq6d','tq7d',
   'psm_fx','psm_fy','psm_fz','psm_tx','psm_ty','psm_tz']

pos_features = ['px','py','pz','qx','qy','qz','qw',
   'vx','vy','vz','wx','wy','wz',
   'q1','q2','q3','q4','q5','q6','q7',
   'vq1','vq2','vq3','vq4','vq5','vq6','vq7',
   'q1d','q2d','q3d','q4d','q5d','q6d','q7d']

#%%
'''
condition_list = [('V',None,False),
                  ('S',None,False),
                  ('S','P',False),
                  ('S','F',False),
                  ('VS',None,False),
                  ('VS','P',False),
                  ('VS','F',False),
                  ('V_RNN',None,True)]'''
condition_list =[('V',None,False)]

if __name__ == "__main__":
    for (model_type,ablate,use_predlist) in condition_list:
        print("model type " + model_type)
        if ablate is not None:
            print("ablation-"+ablate)
        else:
            print("full feature set")
        #model_type = "VS"
        #ablate = "P"
        resnet_type= 50
        feat_extract = False
        force_align = False
        encode = False
        crop_list = []
        #use_predlist = False
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
        for i in range(1,48):
            #crop_list.append((50,350,300,300))
            crop_list.append((270-150,480-150,300,300))
            
        # Define a transformation for the images
        if model_type == "V_RNN":
            trans_function =  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            trans_function = transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
        
        # load the model
        if model_type == "VS":
            model = mdl.StateVisionModel(30, 54, 3,resnet_type=resnet_type,feature_extract=feat_extract,TFN=True)
            model = nn.DataParallel(model)
        elif model_type == "S":
            model  = mdl.StateModel(54, 3)
        elif model_type == "VS_deep":
            model = mdl.StateVisionModel_deep(30, 54, 3)
        elif model_type =="V" or model_type=="V_RNN":
            #model = mdl.VisionModel(3)
            model=mdl.BabyVisionModel()
        
        weight_file =  weight_file = "best_modelweights_" + model_type
        
        if model_type!="S" and feat_extract:
            weight_file="best_modelweights_" + model_type + "_ft"
            
        if force_align and model_type!= "V" :
            weight_file = weight_file + "_faligned"
        
        if resnet_type == 152 and model_type!="S":
            weight_file = weight_file + "_152"
        
        if ablate=="P":
            weight_file = weight_file+"_P"
        elif ablate=="F":
            weight_file = weight_file+"_F"
        
        weight_file = weight_file + "_shallow.dat"
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(weight_file))
        
        # uncomment this to do a encoding using the pre-trained RESNET
        if encode and model_type=="V_RNN":
            model.fc = torch.nn.Tanh()
        elif encode and model_type == "V":
            model.fc = torch.nn.Identity()
            
    
        # load the dataset
        
        file_dir = '../../experiment_data' # define the file directory for dataset
        train_list = [1,3,5,7,
                      8,10,12,14,
                      15,17,19,21,41,42]
        #train_list = [1,3,5,7,43,45]
        val_list = [1]
        config_dict={'file_dir':file_dir,
                 'include_torque': False,
                 'spatial_forces': force_align,
                 'custom_state': None,
                 'batch_size': 16,
                 'crop_list': crop_list,
                 'trans_function': trans_function}
        
    
        
        predictions_list = []
        metrics_list = []
        '''
        test_list_full = [1,3,5,7,8,10,12,14,15,17,19,21]
        condition_list = ['center','center','center','center',
                          'right','right','right','right',
                          'left','left','left','left']
        '''
        '''
        test_list_full = [2,6,9,13,16,20]
        condition_list = ['center','center','right','right','left','left']
        '''
        '''
        test_list_full =  [4,8]
        condition_list = ['center','center']
        '''
        
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
        
        
        '''old normalization'''
        '''
        # compute the max and min force over the entire test set
        max_force = np.array([0.0,0.0,0.0])
        min_force = np.array([np.inf,np.inf,np.inf])
        for test in test_list_full:
            forces = np.loadtxt(file_dir+'/labels_'+str(test)+'.txt',delimiter = ",")[:,1:4]
            test_max = np.max(forces,axis=0)
            test_min = np.min(forces,axis=0)
            for i in range(3):
                if test_max[i] > max_force[i]:
                    print('max force {:f} found, dataset {}, axis {}, time{}'.format(test_max[i],test,i,np.argmax(forces,axis=0)[i]))
                    max_force[i] = test_max[i]
                if test_min[i] < min_force[i]:
                    print('min force {:f} found, dataset {}, axis {}, time{}'.format(test_min[i],test,i,np.argmin(forces,axis=0)[i]))
                    min_force[i] = test_min[i]
        '''
        
        '''normalize over the maximum in that trial'''
        max_force = []
        min_force = []
        for test in test_list_full:
            forces = np.loadtxt(file_dir+'/labels_'+str(test)+'.txt',delimiter = ",")[:,1:4]
            test_max = np.max(forces,axis=0)
            test_min = np.min(forces,axis=0)
            max_force.append(test_max)
            min_force.append(test_min)

        if use_predlist:
            #predlist = pickle.load(open('preds_'+model_type+'_test4.preds',"rb"))
            predlist = pickle.load(open('preds_'+model_type+'.preds',"rb"))
        if encode:
            for i,(test) in enumerate(range(1,48),1):
                test_list = [test]
                loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)
                test_loader = loader_dict['test']
                
                if ablate == "F":
                    mask = np.isin(qty_full,force_features,invert=False)
                    test_loader.dataset.mask_labels(mask)
                if ablate == "P":
                    mask = np.isin(qty_full,pos_features,invert=False)
                    test_loader.dataset.mask_labels(mask)
                
                if use_predlist:
                    predictions = predlist[i]
                else:
                    #plt.close('all')
                    predictions = mdl.evaluate_model(model,test_loader,model_type = model_type,encode=encode)
                    predictions_list.append(predictions)
                
                np.savetxt('encodings_031621/encode_'+str(i+1)+'.txt',predictions) # uncomment to save encodings.
        
        else:
            for i,(test,condition) in enumerate(zip(test_list_full,condition_list),0):
                test_list = [test]
                loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)
                test_loader = loader_dict['test']
                dataset_mean = np.loadtxt('train_dataset_mean.dat')
                dataset_stdev = np.loadtxt('train_dataset_sd.dat')
                
                if ablate == "F":
                    mask = np.isin(qty_full,force_features,invert=False)
                    test_loader.dataset.mask_labels(mask)
                if ablate == "P":
                    mask = np.isin(qty_full,pos_features,invert=False)
                    test_loader.dataset.mask_labels(mask)
                
                if use_predlist:
                    print(i)
                    predictions = predlist[i]
                    predictions[:,:3] = (predictions[:,:3] * dataset_stdev[1:4])+dataset_mean[1:4]
                else:
                    print(condition)
                    #plt.close('all')
                    predictions = mdl.evaluate_model(model,test_loader,model_type = model_type)
                    predictions = (predictions * dataset_stdev[1:4])+dataset_mean[1:4]
                    predictions_list.append(predictions)
               
            # compute the loss and other performance metrics
                if model_type =="V_RNN":
                    #metrics = compute_loss_metrics(predictions[:-10,:],test_loader.dataset.label_array[20:-10,1:4],max_force,min_force,condition,model_type)
                    metrics = compute_loss_metrics(predictions[:,:3],predictions[:,3:],max_force[i],min_force[i],condition,model_type)
                else:
                    metrics = compute_loss_metrics(predictions[10:-10,:],test_loader.dataset.raw_label_array[10:-10,1:4],max_force[i],min_force[i],condition,model_type)
                metrics_list.append(metrics)
            
            # create dataframe
            df_metrics = pd.DataFrame(metrics_list)
            
            import pickle
            df_filedir = 'df_'+model_type
            if ablate == "P":
                df_filedir = df_filedir + "_ablate_P"
            elif ablate == "F":
                df_filedir = df_filedir + "_ablate_F"
            df_filedir = df_filedir + '_test_shallow.df'
            
            pickle.dump(df_metrics,open(df_filedir,'wb'))
            
            if not use_predlist:
                preds_filedir = "preds_"+model_type
                if ablate == "P":
                    preds_filedir = preds_filedir + "_ablate_P"
                elif ablate == "F":
                    preds_filedir = preds_filedir + "_ablate_F"
                preds_filedir += '_shallow.preds'
                pickle.dump(predictions_list,open(preds_filedir,'wb'))
    
                
    #%% Visualization
    
    df_S = pickle.load(open('df_S_test_use_preds.df','rb'))
    df_V = pickle.load(open('df_V_test_use_preds.df','rb'))
    df_V_2 = pickle.load(open('df_V_test_l1-1e-3.df',"rb"))
    df_VS = pickle.load(open('df_VS_test_use_preds.df','rb'))
    df_VS_2 = pickle.load(open('df_VS_test_2.df','rb'))
    df_VS_deep = pickle.load(open('df_VS_deep_test_deep.df','rb'))
    df_VS_152 = pickle.load(open('df_VS_test_152.df','rb'))
    df_VS_152_2 = pickle.load(open('df_VS_test_152-2.df','rb'))
    df_VS_aug100 = pickle.load(open('df_VS_test_aug_100.df','rb'))
    dyn_model_data = pickle.load(open('../dvrk_dynamic_model/dvrk_dynamics_identification/dynamic_model_preds_test.dat','rb'), encoding='latin1')
    df_dyn = pd.DataFrame(dyn_model_data['metric_data'])
    df_VRNN = pickle.load(open('df_V_RNN_test_full4.df','rb'))
    df_VS_F = pickle.load(open('df_VS_test_ablate_F2.df','rb'))
    df_VS_P = pickle.load(open('df_VS_test_ablate_P2.df','rb'))
    df_S_F = pickle.load(open('df_S_test_ablate_F.df','rb'))
    df_S_P = pickle.load(open("df_S_test_ablate_P.df","rb"))
    
    test_numbering_list = [1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,2,2,2]
    df_S['test_number'] = test_numbering_list
    df_VS['test_number'] = test_numbering_list
    df_V['test_number'] = test_numbering_list
    df_dyn['test_number'] = test_numbering_list
    df_VRNN['test_number'] = test_numbering_list
    
    df_VS_aug100['test_number'] = test_numbering_list
    df_VS_aug100['model'] = 'VSaug'
    
    df_VS_deep['test_number'] = test_numbering_list
    df_VS_deep['model'] = 'VSdeep'
    
    df_VS_152['test_number'] =test_numbering_list
    df_VS_152['model'] = 'VS152'
    
    df_VS_152_2['test_number'] = test_numbering_list
    df_VS_152_2['model'] = "VS152-2"
    
    df_V_2['test_number']=test_numbering_list
    df_V_2['model']='V-l1'
    
    df_VS_2['test_number'] = test_numbering_list
    df_VS_2['model']='VS_2'
    
    df_VS_F['test_number'] = test_numbering_list
    df_VS_F['model'] = 'VS_pos_only'
    
    df_VS_P['test_number'] = test_numbering_list
    df_VS_P['model'] = 'VS_force_only'
    
    df_S_F['test_number'] = test_numbering_list
    df_S_F['model'] = 'S_pos_only'
    
    df_S_P['test_number'] = test_numbering_list
    df_S_P['model'] = 'S_force_only' 
    
    df_merge = pd.concat([df_S,df_VS_2,df_V_2,df_dyn,df_VRNN,df_VS_P,df_VS_F,df_S_P,df_S_F])
    #df_merge = pd.concat([df_S,df_VS,df_V])
    #df_merge = pd.concat([df_S,df_VS_152,df_VS_152_2,df_VS])
    #df_merge = pd.melt(df_merge,id_vars=['condition','model'],value_vars=['ME'],var_name = 'metric',value_name='value')
    #df_merge = pd.melt(df_merge,id_vars=['condition','model'],value_vars=['Per Axis RMSEx','Per Axis RMSEy','Per Axis RMSEz'],var_name = 'metric',value_name='value')
    df_merge = pd.melt(df_merge,id_vars=['condition','model','test_number'],value_vars=['Per Axis nRMSEx','Per Axis nRMSEy','Per Axis nRMSEz'],var_name = 'metric',value_name='value')
    df_merge.to_csv("summary_nRMSD3.csv")
    sns.catplot(data=df_merge,x='model',y='value',col='condition',kind='bar')
    #barplot - right to left
    sns.catplot(x='model',y='value',hue="metric",col='condition',col_wrap=3,col_order=['right_less','right','right_more','left_less','left','left_more',],data=df_merge.loc[(df_merge['condition']!='center') & (df_merge['condition']!='new_tool')],kind='bar')
    #barplot - center and new tools/ material
    sns.catplot(x='model',y='value',hue="metric",col='condition',col_wrap=3,data=df_merge.loc[(df_merge['condition']=='center') | (df_merge['condition']=='new_tool')|(df_merge['condition']=='new_material')],kind='bar')
    
    sns.catplot(x='condition',y='value',hue="model",row='metric',order=['right_more','right','right_less','center','left_less','left','left_more'],data=df_merge,kind='point',ci=None)
    sns.catplot(x='condition',y='value',hue="model",row='metric',order=['new_tool','center','new_material'],data=df_merge,kind='point',ci=None,linestyles='-')
    df_merge.groupby(['metric','condition','model']).agg({'value':'mean'})
    
    sns.catplot(x='condition',y='value',hue="model",order=['right_more','right','right_less','center','left_less','left','left_more'],data=df_merge,kind='point',ci=None)
    sns.catplot(x='condition',y='value',hue="model",order=['new_tool','center','new_material'],data=df_merge,kind='point',ci=None,linestyles='-')
    #%%
    
    df_S = pickle.load(open('df_S_val.df','rb'))
    df_V = pickle.load(open('df_V_val.df','rb'))
    df_VS = pickle.load(open('df_VS_val.df','rb'))
    
    test_numbering_list = [1,2,1,2,1,2]
    df_S['test_number'] = test_numbering_list
    df_VS['test_number'] = test_numbering_list
    df_V['test_number'] = test_numbering_list
    
    df_merge = pd.concat([df_S,df_V,df_VS])
    #df_merge = pd.concat([df_S,df_VS])
    #df_merge = pd.melt(df_merge,id_vars=['condition','model','test_number'],value_vars=['Per Axis RMSEx','Per Axis RMSEy','Per Axis RMSEz'],var_name = 'metric',value_name='value')
    df_merge = pd.melt(df_merge,id_vars=['condition','model','test_number'],value_vars=['Per Axis nRMSEx','Per Axis nRMSEy','Per Axis nRMSEz'],var_name = 'metric',value_name='value')
    
    sns.catplot(data=df_merge,x='model',y='value',col='condition',kind='bar')
    #barplot - right to left
    sns.catplot(x='model',y='value',hue="metric",col='condition',col_wrap=3,col_order=['right_less','right','right_more','left_less','left','left_more',],data=df_merge.loc[(df_merge['condition']!='center') & (df_merge['condition']!='new_tool')],kind='bar')
    #barplot - center and new tools/ material
    sns.catplot(x='model',y='value',hue="metric",col='condition',col_wrap=3,data=df_merge.loc[(df_merge['condition']=='center') | (df_merge['condition']=='new_tool')|(df_merge['condition']=='new_material')],kind='bar')
    
    sns.catplot(x='condition',y='value',hue="model",row='metric',order=['right_more','right','right_less','center','left_less','left','left_more'],data=df_merge,kind='point',ci=None)
    sns.catplot(x='condition',y='value',hue="model",row='metric',order=['new_tool','center','new_material'],data=df_merge,kind='point',ci=None,linestyles='-')
    df_merge.groupby(['metric','condition','model']).agg({'value':'mean'})
    
    
    
    #%% do full grid pearson
    plot_order = ['right_more','right','right_less','center','left_less','left','left_more']
    
    #color_dict = {"S":'#1f77b4',"V":'#ff7f0e',"VS":'#2ca02c',"D":'#d62728'}
    #model_type_list = ['S','V','VS','D']
    
    color_dict = {"S":'#1f77b4',"V":'#ff7f0e',"VS":'#d62728'}
    model_type_list = ['S','V','VS']
    
    fig, ax = plt.subplots(3,7,sharex="col",sharey="row")
    axis_namelist= ["X","Y","Z"]
    axis_lims = [(-6,6),(-6,6),(-6,6)]
    
    df_pearson = pd.DataFrame()
    

    for model_type in model_type_list:
        color = color_dict[model_type]
        
        if model_type == 'D':
            predlist = dyn_model_data['pred_data']
        else:
            predlist = pickle.load(open('preds_'+model_type+'_test.preds',"rb"))
            
        for i,(test,condition) in enumerate(zip(test_list_full,condition_list),0):
            if (condition == 'new_tool') or (condition=='new_material'):
                pass
            else:
                continue
            test_list = [test]
            loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)
            test_loader = loader_dict['test']
            
            if model_type == "D":
                predictions = predlist[i]
            else:
                predictions = predlist[i][10:-10,:]
            
            #condition_index = plot_order.index(condition)
            forces = test_loader.dataset.label_array[10:-10,1:4]
            #ax=plot_pearson_grid(ax, condition_index, color , predictions, forces)
            
            df_force = pd.DataFrame(data=forces,columns=["X","Y","Z"])
            df_force['index'] = df_force.index
            df_force = pd.melt(df_force,id_vars=["index"],value_vars=["X","Y","Z"],value_name="Ref.Force",var_name="axis")
            df_pred = pd.DataFrame(predictions,columns=["X","Y","Z"])
            df_pred['index'] = df_pred.index
            df_pred = pd.melt(df_pred,id_vars=["index"],value_vars=["X","Y","Z"],value_name="Pred.Force",var_name="axis")
            df_force = pd.merge(df_force,df_pred,how="left",left_on=["index","axis"],right_on=["index","axis"])
            df_force['condition'] = condition
            df_force['model'] = model_type
            df_pearson = pd.concat((df_pearson,df_force),axis=0)
            
    for i in range(3):
        for j in range(7):
            ax[i,j]=plot_unity_line((-11,11),(-11,11), ax[i,j])
            ax[i,j].set_ylim(axis_lims[i])
            ax[i,j].set_xlim(axis_lims[i])
            #ax[i,j].set_xlim((-6,6))
            ax[i,j].set_xlabel("Ref Force " + axis_namelist[i] + "(N)")
            if j == 0:
                ax[i,j].set_ylabel("Pred Force " + axis_namelist[i] + "(N)")
            if i == 0:
                ax[i,j].title.set_text(plot_order[j])       
    
    sns.lmplot(data= df_pearson,
               x='Ref.Force',
               y='Pred.Force',
               hue='model',
               row="axis",row_order=["X","Y","Z"],
               col='condition',col_order=['right_more','right','right_less','center','left_less','left','left_more'],
               scatter=False,ci=95,x_ci='sd')

#%%
    df_S = pickle.load(open('df_S_test.df','rb'))
    df_S_F = pickle.load(open('df_S_ablate_F_test.df','rb'))
    df_S_P = pickle.load(open("df_S_ablate_P_test.df","rb"))
    df_V = pickle.load(open('df_V_test.df','rb'))
    df_VS_F = pickle.load(open('df_VS_test_ablate_F2.df','rb'))
    df_VS_P = pickle.load(open('df_VS_test_ablate_P2.df','rb'))
    df_VS = pickle.load(open('df_VS_test.df','rb'))
    df_VRNN = pickle.load(open('df_V_RNN_test.df','rb'))
    
    df_S_F['model']='S_F'
    df_S_P['model']='S_P'
    df_VS_F['model']='VS_F'
    df_VS_P['model']='VS_P'
    test_numbering_list = [1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,2,2,2,1,1,1]
    df_S['test_number'] = test_numbering_list
    df_VS['test_number'] = test_numbering_list
    df_V['test_number'] = test_numbering_list
    df_S_P['test_number'] = test_numbering_list
    df_VS_P['test_number'] = test_numbering_list
    df_VS_F['test_number'] = test_numbering_list
    df_S_F['test_number'] = test_numbering_list
    df_VRNN['test_number'] = test_numbering_list
    
    #df_merge = pd.concat([df_S,df_V,df_VS,df_VS_F,df_VS_P,df_S_F,df_S_P,df_VRNN])
    df_merge = pd.concat([df_S,df_S_P,df_S_F,df_VS,df_VS_P,df_VS_F])
    
    df_merge = pd.melt(df_merge,id_vars=['condition','model'],value_vars=['Per Axis RMSEx','Per Axis RMSEy','Per Axis RMSEz'],var_name = 'metric',value_name='value')
    df_merge.to_csv("ablation_nRMSD2.csv")    
    #sns.catplot(x='model',y='value',hue="model",col='condition',row='metric',col_order=['right_more','right','right_less','center','left_less','left','left_more'],data=df_merge,kind='point',ci=None)
    sns.catplot(x='condition',y='value',hue="model",row='metric',data=df_merge,kind='bar',ci=None)
    sns.catplot(x='condition',y='value',hue="model",order=['new_tool','center','new_material'],data=df_merge,kind='point',ci=None,linestyles='-')

    #%%   Manual Plotting
    
    test_list = [2,6,9,13,16,20]  
    loader_dict,loader_sizes = dat.init_dataset(train_list,val_list,test_list,model_type,config_dict)
    test_loader = loader_dict['test']
    if ablate == "F":
        mask = np.isin(qty_full,force_features,invert=False)
        test_loader.dataset.mask_labels(mask)
    if ablate == "P":
        mask = np.isin(qty_full,pos_features,invert=False)
        test_loader.dataset.mask_labels(mask)
    #plt.close('all')
    predictions = mdl.evaluate_model(model,test_loader,model_type = model_type)
    # compute the loss and other performance metrics
    metrics = compute_loss_metrics(predictions,test_loader.dataset.label_array[:,1:4],max_force,min_force,'new_material',"V")
    
    plot_trajectories(predictions,test_loader.dataset.label_array[:,1:4])
    plot_pearson(predictions,test_loader.dataset.label_array[:,1:4])
    gbp_data = compute_GBP(model,test_loader,model_type=model_type)
    plot_heatmaps(gbp_data,predictions,test_loader.dataset.label_array[:,1:4],qty)
    df_gbp_means = compute_and_plot_gbp(gbp_data,qty,False)
    df_gbp_means.groupby('feature').mean().sort_values(by='gbp',ascending=False)
        
