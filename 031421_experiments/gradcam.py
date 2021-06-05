#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:44:14 2021

@author: charm
"""
import dataset_lib as dat
import models as mdl
from torchvision import transforms
from torch.utils import data

import torch 
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import tqdm
import sys
import os

from captum.attr import LayerGradCam

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2

def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap*alpha+img.cpu().numpy().squeeze()
    result = result.div(result.max()).squeeze()

    return heatmap, result

def process_sequence_captum(captum_gc,loader,model_type,device,output_filedir):
    
    try:
        command = 'mkdir ' + output_filedir
        os.system(command)
    except:
        print('file directory exists!')
    
    if model_type !="S":
        mask = [[],[],[]]
        with tqdm.tqdm(total=len(loader)) as pbar:
            for n,(img,state,label) in enumerate(loader,0):
                img = img.to(device,dtype=torch.float)
                if model_type == "VS":
                    state = state.to(device,dtype=torch.float)
                    for axis in range(3):
                        mask[axis] = captum_gc.attribute((img,state),target=axis,relu_attributions=True) 
                else:
                    for axis in range(3):
                        mask[axis] = captum_gc.attribute(img,target=axis,relu_attributions=True)
                results_all = [[],[],[]]
                img = unnorm(img.squeeze())
                for xyz in range(3):
                    upsampled_attr = captum_gc.interpolate(mask[xyz], (224, 224),'bicubic')
                    saliency_map_min, saliency_map_max = upsampled_attr.min(), upsampled_attr.max()
                    upsampled_attr = (upsampled_attr - saliency_map_min).div(saliency_map_max - saliency_map_min).data
                    heatmap, results_all[xyz] = visualize_cam(upsampled_attr, img,alpha=0.5)
                border = np.ones((3,224,5),dtype=np.float32)
                result_xyz = np.concatenate((results_all[0].numpy(),border,results_all[1].numpy(),border,results_all[2].numpy()),axis=2)
                cv2.imwrite(output_filedir+'/heatmap_'+str(n)+'.jpg', (cv2.cvtColor(result_xyz.transpose(1,2,0),cv2.COLOR_RGB2BGR)*255))
                pbar.update(1)
            
    else:
        print('state model has no image...aborting.')

#%%            
unnorm = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

model_type = "VS"
feat_extract = False
force_align = False

crop_list = []
for i in range(1,48):
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
elif model_type == "V":
    model = mdl.VisionModel(3)

weight_file = "best_modelweights_" + model_type

if model_type!="S" and feat_extract:
    weight_file="best_modelweights_" + model_type + "_ft"
    
if force_align and model_type!= "V" :
    weight_file = weight_file + "_faligned"
    
weight_file = weight_file + ".dat"

model.load_state_dict(torch.load(weight_file))

# load the dataset
file_dir = '../../experiment_data' # define the file directory for dataset
config_dict={'file_dir':file_dir,
             'include_torque': False,
             'custom_state': None,
             'batch_size': 1,
             'crop_list': crop_list,
             'spatial_forces': force_align,
             'trans_function': trans_function}


finalconv_name = 'layer4'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#for p in model.parameters():
    #p.requires_grad=True
    
model = model.to(device,dtype=torch.float)
model.eval()
if model_type=="VS":
    target_layer = model.cnn._modules.get(finalconv_name)
elif model_type =="V":
    target_layer = model._modules.get(finalconv_name)
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
'''

test_list_full=[34]
train_list = [1,3,5,7,8,10,12,14,15,17,19,21,41,42]

for test in test_list_full:
    test_list = [test]
    loader_dict,loader_sizes = dat.init_dataset(train_list,train_list,test_list,model_type,config_dict)
    test_loader = loader_dict['test']
    captum_gc = LayerGradCam(model,target_layer)
    output_file = 'heatmaps/imageset_captum' + model_type + "_" + str(test)
    process_sequence_captum(captum_gc,test_loader,model_type,device,output_file)





