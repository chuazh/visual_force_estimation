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

def left_image_callback(msg):
	
	global bridge
	
	image = bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
	
	height,width,channel = image.shape
	rect_h = 300
	rect_w = 300
	
	# draw the crosshairs 
	cv2.rectangle(image, (int(width/2-rect_h/2),int(height/2-rect_h/2)), (int(width/2+rect_h/2),int(height/2+rect_h/2)) , (0,0,255), 3)
	
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
	#cv2.rectangle(image, (int(width/2-rect_h/2),int(height/2-rect_h/2)), (int(width/2+rect_h/2),int(height/2+rect_h/2)) , (0,0,255), 3)
	
	cv2.namedWindow("right", flags= 16)
	cv2.imshow("right",image)
	cv2.moveWindow("right", 3000, 0)
	cv2.setWindowProperty("right", cv2.WND_PROP_FULLSCREEN, 1)
	cv2.waitKey(3)

if __name__ == '__main__':
	
	side = sys.argv[1]
	
	nodename = "crosshair_display_" + side
	rospy.init_node(nodename)
	rate = rospy.Rate(30)
	
	global bridge
	
	bridge = cv_bridge.CvBridge()
	
	if side == 'left':
		left_image_sub = rospy.Subscriber('/camera/left/image_color',Image,left_image_callback)
	elif side == 'right':
		right_image_sub = rospy.Subscriber('/camera/right/image_color',Image,right_image_callback)
	else:
		print('Incorrect side specified!')
	
	rospy.spin()