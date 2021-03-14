#!/usr/bin/env python3

# script to render haptics to dvrk.

import rospy

from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped, TwistStamped
from std_msgs.msg import Float64MultiArray
#import cv2
from cv_bridge import CvBridge

import numpy as np

# all the callback functions are defined here

# PSM1
def cb_PSM1_pos_c(msg):
	global state_input
	state_input[0] = msg.pose.position.x
	state_input[1] = msg.pose.position.y
	state_input[2] = msg.pose.position.z
	state_input[3] = msg.pose.orientation.x
	state_input[4] = msg.pose.orientation.y
	state_input[5] = msg.pose.orientation.z
	state_input[6] = msg.pose.orientation.w

def cb_PSM1_twist(msg):
	global state_input
	state_input[7] = msg.twist.linear.x
	state_input[8] = msg.twist.linear.y
	state_input[9] = msg.twist.linear.z
	state_input[10] = msg.twist.angular.x
	state_input[11] = msg.twist.angular.y
	state_input[12] = msg.twist.angular.z

def cb_PSM1_joint_c(msg):
	global state_input
	state_input[13:19] = msg.position
	state_input[20:26] = msg.velocity
	state_input[27:33] = msg.effort

def cb_PSM1_joint_d(msg):
	global state_input  
	state_input[34:40] = msg.position
	state_input[41:47] = msg.effort

def cb_PSM1_jaw_c(msg):
	global state_input
	state_input[19] = msg.position[0]
	state_input[26] = msg.velocity[0]
	state_input[33] = msg.effort[0]
	
def cb_PSM1_jaw_d(msg):
	global state_input
	state_input[40] = msg.position[0]
	state_input[47] = msg.effort[0]
	
def cb_PSM1_wrench(msg):
	global state_input
	state_input[48]=msg.wrench.force.x
	state_input[49]=msg.wrench.force.y
	state_input[50]=msg.wrench.force.z
	state_input[51]=msg.wrench.torque.x
	state_input[52]=msg.wrench.torque.y
	state_input[53]=msg.wrench.torque.z

# create rosnode
rospy.init_node("haptic_feedback")

# create the state vector for input as a global variable
global state_input
state_input = np.zeros((54,))

# initialize the subscribers
PSM1_pos_c = rospy.Subscriber('/dvrk/PSM2/position_cartesian_current',PoseStamped,cb_PSM1_pos_c)
PSM1_pos_d = rospy.Subscriber('/dvrk/PSM2/position_cartesian_desired',PoseStamped,)
PSM1_joint_c = rospy.Subscriber('/dvrk/PSM2/state_joint_current',JointState,cb_PSM1_joint_c)
#PSM1_joint_d = rospy.Subscriber('/dvrk/PSM2/state_joint_desired',JointState,cb_PSM1_joint_d)
PSM1_jaw_c = rospy.Subscriber('/dvrk/PSM2/state_jaw_current',JointState,cb_PSM1_jaw_c)
PSM1_jaw_d = rospy.Subscriber('/dvrk/PSM2/state_jaw_desired',JointState,cb_PSM1_jaw_d)
PSM1_twist = rospy.Subscriber('/dvrk/PSM2/twist_body_current',TwistStamped,cb_PSM1_twist)
PSM1_wrench = rospy.Subscriber('/dvrk/PSM2/wrench_body_current',WrenchStamped,cb_PSM1_wrench)

state_publisher = rospy.Publisher('ml_state_input',Float64MultiArray,queue_size = 1)
state_multarray = Float64MultiArray()

r = rospy.Rate(1000)
while not rospy.is_shutdown():
	state_multarray.data = list(state_input)
	state_publisher.publish(state_multarray)
	r.sleep()
