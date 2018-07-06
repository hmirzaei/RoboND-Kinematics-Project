from sympy import *
from time import time
from mpmath import radians
import tf

'''
Format of test case is [ [[EE position],[EE orientation as quaternions]],[WC location],[joint angles]]
You can generate additional test cases by setting up your kuka project and running `$ roslaunch kuka_arm forward_kinematics.launch`
From here you can adjust the joint angles to find thetas, use the gripper to extract positions and orientation (in quaternion xyzw) and lastly use link 5
to find the position of the wrist center. These newly generated test cases can be added to the test_cases dictionary.
'''

test_cases = {1:[[[2.16135,-1.42635,1.55109],
                  [0.708611,0.186356,-0.157931,0.661967]],
                  [1.89451,-1.44302,1.69366],
                  [-0.65,0.45,-0.36,0.95,0.79,0.49]],
              2:[[[-0.56754,0.93663,3.0038],
                  [0.62073, 0.48318,0.38759,0.480629]],
                  [-0.638,0.64198,2.9988],
                  [-0.79,-0.11,-2.33,1.94,1.14,-3.68]],
              3:[[[-1.3863,0.02074,0.90986],
                  [0.01735,-0.2179,0.9025,0.371016]],
                  [-1.1669,-0.17989,0.85137],
                  [-2.99,-0.12,0.94,4.06,1.29,-4.12]],
              4:[],
              5:[]}


def test_code(test_case):
    ## Set up code
    ## Do not modify!
    x = 0
    class Position:
        def __init__(self,EE_pos):
            self.x = EE_pos[0]
            self.y = EE_pos[1]
            self.z = EE_pos[2]
    class Orientation:
        def __init__(self,EE_ori):
            self.x = EE_ori[0]
            self.y = EE_ori[1]
            self.z = EE_ori[2]
            self.w = EE_ori[3]

    position = Position(test_case[0][0])
    orientation = Orientation(test_case[0][1])

    class Combine:
        def __init__(self,position,orientation):
            self.position = position
            self.orientation = orientation

    comb = Combine(position,orientation)

    class Pose:
        def __init__(self,comb):
            self.poses = [comb]

    req = Pose(comb)
    start_time = time()
    
    ########################################################################################
    ## 

    ## Insert IK code here!
    import numpy as np
    
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

    ## 
    ########################################################################################
    
    ########################################################################################
    ## For additional debugging add your forward kinematics here. Use your previously calculated thetas
    ## as the input and output the position of your end effector as your_ee = [x,y,z]

    ## (OPTIONAL) YOUR CODE HERE!

    ## End your code input for forward kinematics here!
    ########################################################################################

    ## For error analysis please set the following variables of your WC location and EE location in the format of [x,y,z]
    your_wc = [1,1,1] # <--- Load your calculated WC values in this array
    your_ee = [1,1,1] # <--- Load your calculated end effector value from your forward kinematics
    ########################################################################################

    ## Error analysis
    print ("\nTotal run time to calculate joint angles from pose is %04.4f seconds" % (time()-start_time))

    # Find WC error
    if not(sum(your_wc)==3):
        wc_x_e = abs(your_wc[0]-test_case[1][0])
        wc_y_e = abs(your_wc[1]-test_case[1][1])
        wc_z_e = abs(your_wc[2]-test_case[1][2])
        wc_offset = sqrt(wc_x_e**2 + wc_y_e**2 + wc_z_e**2)
        print ("\nWrist error for x position is: %04.8f" % wc_x_e)
        print ("Wrist error for y position is: %04.8f" % wc_y_e)
        print ("Wrist error for z position is: %04.8f" % wc_z_e)
        print ("Overall wrist offset is: %04.8f units" % wc_offset)

    # Find theta errors
    t_1_e = abs(theta1-test_case[2][0])
    t_2_e = abs(theta2-test_case[2][1])
    t_3_e = abs(theta3-test_case[2][2])
    t_4_e = abs(theta4-test_case[2][3])
    t_5_e = abs(theta5-test_case[2][4])
    t_6_e = abs(theta6-test_case[2][5])
    print ("\nTheta 1 error is: %04.8f" % t_1_e)
    print ("Theta 2 error is: %04.8f" % t_2_e)
    print ("Theta 3 error is: %04.8f" % t_3_e)
    print ("Theta 4 error is: %04.8f" % t_4_e)
    print ("Theta 5 error is: %04.8f" % t_5_e)
    print ("Theta 6 error is: %04.8f" % t_6_e)
    print ("\n**These theta errors may not be a correct representation of your code, due to the fact \
           \nthat the arm can have muliple positions. It is best to add your forward kinmeatics to \
           \nconfirm whether your code is working or not**")
    print (" ")

    # Find FK EE error
    if not(sum(your_ee)==3):
        ee_x_e = abs(your_ee[0]-test_case[0][0][0])
        ee_y_e = abs(your_ee[1]-test_case[0][0][1])
        ee_z_e = abs(your_ee[2]-test_case[0][0][2])
        ee_offset = sqrt(ee_x_e**2 + ee_y_e**2 + ee_z_e**2)
        print ("\nEnd effector error for x position is: %04.8f" % ee_x_e)
        print ("End effector error for y position is: %04.8f" % ee_y_e)
        print ("End effector error for z position is: %04.8f" % ee_z_e)
        print ("Overall end effector offset is: %04.8f units \n" % ee_offset)




if __name__ == "__main__":
    # Change test case number for different scenarios
    test_case_number = 1

    test_code(test_cases[test_case_number])
