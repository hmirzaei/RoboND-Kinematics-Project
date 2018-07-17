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
        
        import numpy as np
        from numpy import array
        from sympy import symbols, cos, sin, pi, simplify, sqrt, atan2
        from sympy.matrices import Matrix
        import math

        ## symbols of joint variables
        q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8') #theta_i
        d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
        a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')
        
        # DH Parameters
        s = {alpha0:     0, a0:      0, d1:  0.75,
             alpha1: -pi/2, a1:   0.35, d2:     0,   q2: q2 - pi/2,
             alpha2:     0, a2:   1.25, d3:     0,
             alpha3: -pi/2, a3: -0.054, d4:  1.50,
             alpha4:  pi/2, a4:      0, d5:     0,
             alpha5: -pi/2, a5:      0, d6:     0,
             alpha6:     0, a6:      0, d7: 0.303,   q7: 0}
        
        #### Homogeneous Transforms
        # base_link to link1
        T0_1 = Matrix([[             cos(q1),            -sin(q1),            0,              a0],
                       [ sin(q1)*cos(alpha0), cos(q1)*cos(alpha0), -sin(alpha0), -sin(alpha0)*d1],
                       [ sin(q1)*sin(alpha0), cos(q1)*sin(alpha0),  cos(alpha0),  cos(alpha0)*d1],
                       [                   0,                   0,            0,               1]])
        T0_1 = T0_1.subs(s)
        
        #link1 to link2
        T1_2 = Matrix([[             cos(q2),            -sin(q2),            0,              a1],
                       [ sin(q2)*cos(alpha1), cos(q2)*cos(alpha1), -sin(alpha1), -sin(alpha1)*d2],
                       [ sin(q2)*sin(alpha1), cos(q2)*sin(alpha1),  cos(alpha1),  cos(alpha1)*d2],
                       [                   0,                   0,            0,               1]])
        T1_2 = T1_2.subs(s)
        
        #link2 to link3
        T2_3 = Matrix([[             cos(q3),            -sin(q3),            0,              a2],
                       [ sin(q3)*cos(alpha2), cos(q3)*cos(alpha2), -sin(alpha2), -sin(alpha2)*d3],
                       [ sin(q3)*sin(alpha2), cos(q3)*sin(alpha2),  cos(alpha2),  cos(alpha2)*d3],
                       [                   0,                   0,            0,               1]])
        T2_3 = T2_3.subs(s)
        
        #link3 to link4
        T3_4 = Matrix([[             cos(q4),            -sin(q4),            0,              a3],
                       [ sin(q4)*cos(alpha3), cos(q4)*cos(alpha3), -sin(alpha3), -sin(alpha3)*d4],
                       [ sin(q4)*sin(alpha3), cos(q4)*sin(alpha3),  cos(alpha3),  cos(alpha3)*d4],
                       [                   0,                   0,            0,               1]])
        T3_4 = T3_4.subs(s)
        
        #link4 to link5
        T4_5 = Matrix([[             cos(q5),            -sin(q5),            0,              a4],
                       [ sin(q5)*cos(alpha4), cos(q5)*cos(alpha4), -sin(alpha4), -sin(alpha4)*d5],
                       [ sin(q5)*sin(alpha4), cos(q5)*sin(alpha4),  cos(alpha4),  cos(alpha4)*d5],
                       [                   0,                   0,            0,               1]])
        T4_5 = T4_5.subs(s)
        
        #link5 to link6
        T5_6 = Matrix([[             cos(q6),            -sin(q6),            0,              a5],
                       [ sin(q6)*cos(alpha5), cos(q6)*cos(alpha5), -sin(alpha5), -sin(alpha5)*d6],
                       [ sin(q6)*sin(alpha5), cos(q6)*sin(alpha5),  cos(alpha5),  cos(alpha5)*d6],
                       [                   0,                   0,            0,               1]])
        T5_6 = T5_6.subs(s)
        
        #link6 to gripper frame
        T6_G = Matrix([[             cos(q7),            -sin(q7),            0,              a6],
                       [ sin(q7)*cos(alpha6), cos(q7)*cos(alpha6), -sin(alpha6), -sin(alpha6)*d7],
                       [ sin(q7)*sin(alpha6), cos(q7)*sin(alpha6),  cos(alpha6),  cos(alpha6)*d7],
                       [                   0,                   0,            0,               1]])
        T6_G = T6_G.subs(s)
        
        # Corerection rotation matrix for the difference between DH and urdf reference frames for the gripper link
        R_z = Matrix([[     cos(np.pi), -sin(np.pi),             0,   0],
                      [     sin(np.pi),  cos(np.pi),             0,   0],
                      [              0,           0,             1,   0],
                      [              0,           0,             0,   1]])
         
        R_y = Matrix([[  cos(-np.pi/2),           0, sin(-np.pi/2),   0],
                      [              0,           1,             0,   0],
                      [ -sin(-np.pi/2),           0, cos(-np.pi/2),   0],
                      [              0,           0,       0,         1]])
        R_corr = R_z * R_y
        
        
        # Homogeneous transformation from base link to gripper frame 
        T0_G = simplify(T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_G * R_corr)
        
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
