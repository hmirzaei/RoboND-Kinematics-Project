#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
import numpy as np


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        ### Your FK code here
        # Create symbols
	#
	#
	# Create Modified DH parameters
	#
	#
	# Define Modified DH Transformation matrix
	#
	#
	# Create individual transformation matrices
	#
	#
	# Extract rotation matrices from the transformation matrices
	#
	#
        ###

        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

	    # Extract end-effector position and orientation from request
	    # px,py,pz = end-effector position
	    # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            # (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            #     [req.poses[x].orientation.x, req.poses[x].orientation.y,
            #         req.poses[x].orientation.z, req.poses[x].orientation.w])

            ### Your IK code here

            # all the calculation are done in Rviz coordinate frames to avoid extra transformations
            # Rotation Matrix from base to gripper
            R_bg = tf.transformations.quaternion_matrix(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                     req.poses[x].orientation.z, req.poses[x].orientation.w])[:3, :3]

            # calculate wrist center
            wc = [req.poses[x].position.x, req.poses[x].position.y, req.poses[x].position.z] - (0.193 + 0.11) * R_bg[:3,0]

            # project wrist center and link2's origin to the plane passing through link2's origin and perpenduclar to horizontal plane  
            wc_x = np.sqrt(wc[0]**2+wc[1]**2)
            wc_y = wc[2]
            l2_x = 0.35
            l2_y = 0.33 + 0.42

            # find the edge lengths of the triangle made by wriste center, link2's and link3's origins.
            l2_l3 = 1.25
            l2_wc = np.sqrt((wc_x - l2_x)**2 + (wc_y - l2_y)**2)
            l3_wc = np.sqrt((0.96+0.54)**2+0.054**2)
            
            # calculate the first three joint angles
            theta1 = np.arctan2(wc[1], wc[0])
            theta2 = np.pi/2 - np.arctan2(wc_y - l2_y, wc_x - l2_x) - np.arccos((l2_wc**2+l2_l3**2-l3_wc**2)/(2*l2_wc*l2_l3))
            theta3 = np.pi/2 - np.arctan2(0.054, 0.96+0.54) - np.arccos((l3_wc**2+l2_l3**2-l2_wc**2)/(2*l3_wc*l2_l3))

            # calculate rotation matrix of the first three joints
            R_z = Matrix([[   np.cos(theta1),-np.sin(theta1),             0],
                          [   np.sin(theta1), np.cos(theta1),             0],
                          [             0,           0,             1]])

            R_y =  Matrix([[  np.cos(theta2+theta3),           0, np.sin(theta2+theta3)],
                           [              0,           1,             0],
                           [ -np.sin(theta2+theta3),           0, np.cos(theta2+theta3)]])

            R_b3 = np.array((R_z * R_y).tolist(), np.float)


            # calculate the rotation matrix from link3 to the gripper using the whole rotation matrix and rotation matrix
            # from base to link 3
            R_3g = np.array((Matrix(R_b3).T * R_bg).tolist(), float)

            # calculate the first three joint angles
            theta4 = np.arctan2(R_3g[1, 0], -R_3g[2, 0])
            theta5 = np.arctan2(np.sqrt(R_3g[1, 0] ** 2 + R_3g[2, 0] ** 2), R_3g[0, 0])
            theta6 = np.arctan2(R_3g[0,1], R_3g[0,2])

            
            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
	    joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
	    joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
