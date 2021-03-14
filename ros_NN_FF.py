#!/usr/bin/env python3

import rospy
import torch
import numpy as np
import torch.nn as nn
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import WrenchStamped, Wrench
from sensor_msgs.msg import Joy
import models as mdl
import scipy.signal as sig

def state_callback(msg):
	global state_input
	global force_pred
	global force_pred_out
	global mean1
	global std1
	
	state_input= np.array(msg.data)
	state_input = (state_input - mean1[7:61])/std1[7:61]

def teleop_callback(data):
	global teleop
	butt = data.buttons[0]
	if butt > 0.5:
		teleop = True
	else:
		teleop = False

#load model
model = mdl.StateModel(54,3)
print("loading the model...")
weight_file = "best_modelweights_S.dat" #"best_modelweights_S_PSM1.dat"
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

global teleop
teleop = False

state_input = np.zeros((54,))

state_subscriber=rospy.Subscriber('ml_state_input',Float64MultiArray,state_callback,queue_size=1)
teleop_subscriber = rospy.Subscriber('/dvrk/footpedals/coag',Joy, teleop_callback,queue_size=1)

force_pred_out = WrenchStamped()

publisher = rospy.Publisher('nn_force_pred',WrenchStamped, queue_size=1)

publisher2 = rospy.Publisher('dvrk/MTMR/set_wrench_body',Wrench,queue_size=1) # NOTE THAT THE BODY FORCES ARE NOT SET TO SPATIAL FRAME!!
force_pred_out2 = Wrench()

mean1 = np.loadtxt('PSM2_mean.csv')
std1 = np.loadtxt('PSM2_std.csv')

# create butterworth filter parameters
order = 3
fc = 5.0
fs = 1000
b,a = sig.butter(order,fc/(fs/2.0))
print(b)
print(a)
print(" ")

# create list of old variables needed for difference filter
force_pred_old = [np.zeros((3,)),np.zeros((3,)),np.zeros((3,))]
filtered_force_old = [np.zeros((3,)),np.zeros((3,)),np.zeros((3,))]

print("starting inference...")
r = rospy.Rate(1000)

'''
rate_list = []
previous_time = rospy.get_time()
print_time = previous_time'''

while not rospy.is_shutdown():

	'''
	current_time = rospy.get_time()
	rate_list.append(1.0/(current_time-previous_time))
	previous_time = current_time
	
	if current_time - print_time > 1:
		print(np.mean(rate_list))
		rate_list = []
		print_time = current_time
	'''

	state_input_gpu = torch.from_numpy(state_input).to(device,torch.float)
	state_input_gpu = state_input_gpu.unsqueeze(0)

	force_pred = model(state_input_gpu).squeeze().cpu().detach().numpy()
	force_pred = (force_pred * std1[1:4]) + mean1[1:4]
	
	
	force_pred_out.wrench.force.x = -force_pred[0]
	force_pred_out.wrench.force.y = -force_pred[1]
	force_pred_out.wrench.force.z = -force_pred[2]
	force_pred_out2.force.x = force_pred_out.wrench.force.x
	force_pred_out2.force.y = force_pred_out.wrench.force.y
	force_pred_out2.force.z = force_pred_out.wrench.force.z
	
	'''
	force_pred_out.wrench.force.x = -a[1]*filtered_force_old[0][0] -a[2]*filtered_force_old[1][0] -a[3]*filtered_force_old[2][0] + b[0] * force_pred[0] + b[1]*force_pred_old[0][0] + b[2]*force_pred_old[1][0] + b[3]*force_pred_old[2][0]
	force_pred_out.wrench.force.y =  -a[1]*filtered_force_old[0][1] -a[2]*filtered_force_old[1][1] -a[3]*filtered_force_old[2][1] + b[0] * force_pred[1] + b[1]*force_pred_old[0][1] + b[2]*force_pred_old[1][1] + b[3]*force_pred_old[2][1]
	force_pred_out.wrench.force.z =  -a[1]*filtered_force_old[0][2] -a[2]*filtered_force_old[1][2] -a[3]*filtered_force_old[2][2] + b[0] * force_pred[2] + b[1]*force_pred_old[0][2] + b[2]*force_pred_old[1][2] + b[3]*force_pred_old[2][2]
	force_pred_out2.force.x = -force_pred_out.wrench.force.x
	force_pred_out2.force.y = -force_pred_out.wrench.force.y
	force_pred_out2.force.z = -force_pred_out.wrench.force.z
	
	force_pred_old[2] = force_pred_old[1]
	force_pred_old[1] = force_pred_old[0]
	force_pred_old[0] = force_pred
	filtered_force_old[2] = filtered_force_old[1]
	filtered_force_old[1] = filtered_force_old[0]
	filtered_force_old[0] = np.array([force_pred_out.wrench.force.x,force_pred_out.wrench.force.y,force_pred_out.wrench.force.z])
	'''
	#publisher.publish(force_pred_out)
	
	if teleop:
		publisher2.publish(force_pred_out2)
	
	r.sleep()