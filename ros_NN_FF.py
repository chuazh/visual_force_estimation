#!/usr/bin/env python3

import rospy
import torch
import numpy as np
import torch.nn as nn
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import WrenchStamped
import models as mdl

def state_callback(msg):
	global state_input
	global force_pred
	global force_pred_out
	global mean1
	global std1
	
	state_input= np.array(msg.data)
	state_input = (state_input - mean1[7:61])/std1[7:61]
	state_input_gpu = torch.from_numpy(state_input).to(device,torch.float)
	state_input_gpu = state_input_gpu.unsqueeze(0)
	
	force_pred = model(state_input_gpu).squeeze().cpu().detach().numpy()
	
	force_pred = (force_pred * std1[1:4]) + mean1[1:4]
	
	force_pred_out.wrench.force.x = force_pred[0]
	force_pred_out.wrench.force.y = force_pred[1]
	force_pred_out.wrench.force.z = force_pred[2]
	

#load model
model = mdl.StateModel(54,3)
print("loading the model...")
weight_file = "best_modelweights_S_PSM1.dat"
model.load_state_dict(torch.load(weight_file))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device,torch.float)
model.eval()

# create rosnode
rospy.init_node("NN_node")
# create the state vector for input as a global variable
global state_input
global force_pred
global force_pred_out
global mean1
global std1

state_input = np.zeros((54,))

state_subscriber=rospy.Subscriber('ml_state_input',Float64MultiArray,state_callback,queue_size=1)
force_pred_out = WrenchStamped()

publisher = rospy.Publisher('nn_force_pred',WrenchStamped, queue_size=1)

mean1 = np.loadtxt('PSM1_mean.csv')
std1 = np.loadtxt('PSM1_std.csv')

print("starting inference...")
r = rospy.Rate(1000)
while not rospy.is_shutdown():	
	publisher.publish(force_pred_out)
	r.sleep()