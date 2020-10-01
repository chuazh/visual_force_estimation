#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:58:05 2020

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

import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2

class GradCAM:
    """Calculate GradCAM salinecy map.
    Args:
        input: input image with shape of (1, 3, H, W)
        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
    Return:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
        self.model_arch = arch

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def saliency_map_size(self, *input_size):
        device = next(self.model_arch.parameters()).device
        self.model_arch(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        
        image,state = input
        b, c, h, w = image.size()

        output = self.model_arch(image,state)
        scores = [output[:,0].squeeze(),output[:,1].squeeze(),output[:,2].squeeze()]
        saliency_maps = [[],[],[]]
        for score,n in zip(scores,range(3)):
            self.model_arch.zero_grad()
            score.backward(retain_graph=retain_graph)
            gradients = self.gradients['value']
            activations = self.activations['value']
            b, k, u, v = gradients.size()

            alpha = gradients.view(b, k, -1).mean(2)
            # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
            weights = alpha.view(b, k, 1, 1)

            saliency_map = (weights*activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            saliency_maps[n] = saliency_map

        return saliency_maps, output

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


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


if __name__ == "__main__":

    model_type = "VS"
    feat_extract = True
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
    for i in range(1,34):
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
    
    weight_file =  weight_file = "best_modelweights_" + model_type
    
    if model_type!="S" and feat_extract:
        weight_file="best_modelweights_" + model_type + "_ft"
        
    if force_align and model_type!= "V" :
        weight_file = weight_file + "_faligned"
        
    weight_file = weight_file + ".dat"
    
    model.load_state_dict(torch.load(weight_file))
    
    # load the dataset
    file_dir = '../experiment_data' # define the file directory for dataset
    train_list = [1,2,3,4,5,7,8]
    test_list = [2,6,9,13,16,20]
    config_dict={'file_dir':file_dir,
                 'include_torque': False,
                 'custom_state': None,
                 'batch_size': 1,
                 'crop_list': crop_list,
                 'spatial_forces': force_align,
                 'trans_function': trans_function}
    
    loader_dict,loader_sizes = dat.init_dataset(train_list,test_list,test_list,model_type,config_dict)
    test_loader = loader_dict['test']
    
    unnorm = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
    
    finalconv_name = 'layer4'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for p in model.parameters():
        p.requires_grad=True
        
    model = model.to(device,dtype=torch.double)
    model.eval()
    target_layer = model.cnn._modules.get(finalconv_name)
    gradcam = GradCAM(model, target_layer)
    #%%
    inputs,inputs_aug,_ = next(iter(loader_dict['test']))
   
    mask, _ = gradcam((inputs.to(device,dtype=torch.double),inputs_aug.to(device,dtype=torch.double)),class_idx=None,retain_graph=True)  
    results_all = [[],[],[]]
    heatmap = [[],[],[]]
    for xyz in range(3):
        img = unnorm(inputs.squeeze())
        heatmap[xyz], results_all[xyz] = visualize_cam(mask[xyz], img,alpha=0.5)
        
    plt.imshow(results_all[0].cpu().detach().numpy().transpose(1,2,0))
