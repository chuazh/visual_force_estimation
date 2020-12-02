#!/usr/bin/env python

'''
ROS Node to with rectangular overlay so that we know where to aim our camera...
'''

import rospy
import cv2, cv_bridge
import argparse
import numpy as np
import sys

from sensor_msgs.msg import Image
from geometry_msgs.msg import WrenchStamped

def left_image_callback(msg):
	
	global bridge
	
	image = bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
	
	height,width,channel = image.shape
	rect_h = 300
	rect_w = 300
	
	# draw the crosshairs 
	cv2.rectangle(image, (int(width/2-rect_h/2),int(height/2-rect_h/2)), (int(width/2+rect_h/2),int(height/2+rect_h/2)) , (0,0,255), 3)
	# draw the predicted forces from the neural net
	cv2.putText(image,'X: '+str(round(nn_force_pred[0],2)),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
	cv2.putText(image,'Y: '+str(round(nn_force_pred[1],2)),(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
	cv2.putText(image,'Z: '+str(round(nn_force_pred[2],2)),(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
	
	cv2.namedWindow("left", flags= 16)
	cv2.imshow("left",image)
	cv2.moveWindow("left", 2000, 0)
	cv2.setWindowProperty("left", cv2.WND_PROP_FULLSCREEN, 1) 
	cv2.waitKey(3)
	
def right_image_callback(msg):
	
	global bridge
	
	image = bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
	
	height,width,channel = image.shape
	rect_h = 300
	rect_w = 300
	
	# draw the crosshairs 
	cv2.rectangle(image, (int(width/2-rect_h/2),int(height/2-rect_h/2)), (int(width/2+rect_h/2),int(height/2+rect_h/2)) , (0,0,255), 3)
	# draw the predicted forces from the neural net
	cv2.putText(image,'X: '+str(round(nn_force_pred[0],2)),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
	cv2.putText(image,'Y: '+str(round(nn_force_pred[1],2)),(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
	cv2.putText(image,'Z: '+str(round(nn_force_pred[2],2)),(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
	
	cv2.namedWindow("right", flags= 16)
	cv2.imshow("right",image)
	cv2.moveWindow("right", 3000, 0)
	cv2.setWindowProperty("right", cv2.WND_PROP_FULLSCREEN, 1)
	cv2.waitKey(3)
	
def nn_pred_callback(msg):
	global nn_force_pred
	nn_force_pred[0] = msg.wrench.force.x
	nn_force_pred[1] = msg.wrench.force.y
	nn_force_pred[2] = msg.wrench.force.z

if __name__ == '__main__':
	
	side = sys.argv[1]
	
	nodename = "crosshair_display_" + side
	rospy.init_node(nodename)
	rate = rospy.Rate(30)
	
	global bridge
	global nn_force_pred
	
	nn_force_pred = np.zeros((3,))
	
	bridge = cv_bridge.CvBridge()
	
	if side == 'left':
		left_image_sub = rospy.Subscriber('/camera/left/image_color',Image,left_image_callback)
	elif side == 'right':
		right_image_sub = rospy.Subscriber('/camera/right/image_color',Image,right_image_callback)
	else:
		print('Incorrect side specified!')
		
	nn_pred_sub = rospy.Subscriber('/nn_force_pred',WrenchStamped,nn_pred_callback,queue_size=None)
	
	rospy.spin()