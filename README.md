[//]: # (Image References)
[image_0]: ./misc/rover_image.jpg
[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)
# Project: Kinematics Pick & Place

### Simulation Output
[run_gif]: ./output/run.gif
![run_gif][run_gif]

### Kinematic Analysis
#### DH Reference Frame Diagram
[dh]: ./misc/dh.png
![dh][dh]

Denavit–Hartenberg reference frames for KUKA robot arm is shown in above diagram. The following pictures show the actual joints and their reference frames defined in RViz environment

[l1-3]: ./misc/l1-3.png
[l4-6]: ./misc/l4-6.png

| ![l1-3][l1-3] | ![l4-6][l4-6] |
|:-------------:|:-------------:|
| joints 1 to 3 | joints 4 - 6 |

Using above diagrams we can derive DH parameter table as following:

i   | α<sub>i-1</sub> | a<sub>i-1</sub>      | d<sub>i-1</sub>    | θ<sub>i</sub>
:---|:----------------|:---------------------|:-------------------|:---------------
1   | 0               | 0                    | d<sub>1</sub>=0.75 | θ<sub>1</sub>
2   | - π/2           | a<sub>1</sub>=0.35   | 0                  | θ<sub>2</sub> - π/2
3   | 0               | a<sub>2</sub>=1.25   | 0                  | θ<sub>3</sub>
4   | - π/2           | a<sub>3</sub>=-0.054 | d<sub>4</sub>=1.5  | θ<sub>4</sub>
5   | π/2             | 0                    | 0                  | θ<sub>5</sub> 
6   | - π/2           | 0                    | 0                  | θ<sub>6</sub>
G*  | 0               | 0                    | d<sub>G</sub>=0.303| 0

\*  gripper 

To find the actual values for the link lengths and link offsets, we can refer to `kr210.urdf.xacro` located in urdf directory. The following table summarizes the entries in this file:

[dh_vals]: ./misc/dh_vals.png
![dh_vals][dh_vals]

#### Forward kinematics

Using `sympy` python package we can derive forward kinematic equations. The instructions provided in the lectures are followed to develop the python codes.

First, we should import the required functions and packages:
```python
import numpy as np
from numpy import array
from sympy import symbols, cos, sin, pi, simplify, sqrt, atan2
from sympy.matrices import Matrix
import math
```

Next, the symbolic variables are defined for joint angles and DH parameters:
```python
### symbols of joint variables
q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8') #theta_i
d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')
```

DH parameters are populated using the DH table described before:
```python
# DH Parameters
s = {alpha0:     0, a0:      0, d1:  0.75,
     alpha1: -pi/2, a1:   0.35, d2:     0,   q2: q2 - pi/2,
     alpha2:     0, a2:   1.25, d3:     0,
     alpha3: -pi/2, a3: -0.054, d4:  1.50,
     alpha4:  pi/2, a4:      0, d5:     0,
     alpha5: -pi/2, a5:      0, d6:     0,
     alpha6:     0, a6:      0, d7: 0.303,   q7: 0}
```
Next, the transformation matrices from each link to the subsequent link are calculated:
```python
#### Homogeneous Transforms
# base_link to link1
T0_1 = Matrix([[             cos(q1),            -sin(q1),            0,              a0],
               [ sin(q1)*cos(alpha0), cos(q1)*cos(alpha0), -sin(alpha0), -sin(alpha0)*d1],
               [ sin(q1)*sin(alpha0), cos(q1)*sin(alpha0),  cos(alpha0),  cos(alpha0)*d1],
               [                   0,                   0,            0,               1]])
T0_1 = T0_1.subs(s)

# link1 to link2
T1_2 = Matrix([[             cos(q2),            -sin(q2),            0,              a1],
               [ sin(q2)*cos(alpha1), cos(q2)*cos(alpha1), -sin(alpha1), -sin(alpha1)*d2],
               [ sin(q2)*sin(alpha1), cos(q2)*sin(alpha1),  cos(alpha1),  cos(alpha1)*d2],
               [                   0,                   0,            0,               1]])
T1_2 = T1_2.subs(s)

#...

T5_6 = Matrix([[             cos(q6),            -sin(q6),            0,              a5],
               [ sin(q6)*cos(alpha5), cos(q6)*cos(alpha5), -sin(alpha5), -sin(alpha5)*d6],
               [ sin(q6)*sin(alpha5), cos(q6)*sin(alpha5),  cos(alpha5),  cos(alpha5)*d6],
               [                   0,                   0,            0,               1]])
T5_6 = T5_6.subs(s)

# link6 to gripper
T6_G = Matrix([[             cos(q7),            -sin(q7),            0,              a6],
               [ sin(q7)*cos(alpha6), cos(q7)*cos(alpha6), -sin(alpha6), -sin(alpha6)*d7],
               [ sin(q7)*sin(alpha6), cos(q7)*sin(alpha6),  cos(alpha6),  cos(alpha6)*d7],
               [                   0,                   0,            0,               1]])
T6_G = T6_G.subs(s)
```

To account for the difference between gripper frame in DH reference frame and urdf reference frame, a correction rotation matrix is defined to align these two reference frames:
```python
# Correction rotation matrix for the difference between DH and urdf reference frames for the gripper link
R_z = Matrix([[     cos(np.pi), -sin(np.pi),             0,   0],
              [     sin(np.pi),  cos(np.pi),             0,   0],
              [              0,           0,             1,   0],
              [              0,           0,             0,   1]])
 
R_y = Matrix([[  cos(-np.pi/2),           0, sin(-np.pi/2),   0],
              [              0,           1,             0,   0],
              [ -sin(-np.pi/2),           0, cos(-np.pi/2),   0],
              [              0,           0,       0,         1]])
R_corr = R_z * R_y
```

Now, we can derive the total transformation from the base link to gripper link by multiplying all the calculated matrices:
```python
# Homogeneous transformation from base link to gripper frame 
T0_G = simplify(T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_G * R_corr)
```

To verify the forward kinematic calculations, the symbolic joint angles are substituted by the first test case and the results are printed:
```python
# first test case
q1_v = -0.65
q2_v = 0.45
q3_v = -0.36
q4_v = 0.95
q5_v = 0.79
q6_v = 0.49

# substitute symbolic variables by test case values
T0_G_val = T0_G.evalf(subs={q1: q1_v, q2: q2_v, q3: q3_v, q4: q4_v, q5: q5_v, q6: q6_v})
T0_G_val = np.array(T0_G_val.tolist(), np.float)
print("Transformation Matrix from base to end effector= ")
print(T0_G_val)
print()
print("Translation = ", T0_G_val[:3,3])
print("roll = ", np.arctan2(T0_G_val[2,1], T0_G_val[2,2])) #roll
print("pitch = ", np.arctan2(-T0_G_val[2,0], np.sqrt(T0_G_val[0,0]**2 + T0_G_val[1,0]**2))) #pitch
print("yaw = ",np.arctan2(T0_G_val[1,0], T0_G_val[0,0])) #yaw
```
Results:
```
Transformation Matrix from base to end effector= 
[[ 0.87817143  0.47774295  0.02401277  2.16298055]
 [ 0.05822874 -0.05693817 -0.99667821 -1.42438431]
 [-0.47478875  0.87665256 -0.07781984  1.54309862]
 [ 0.          0.          0.          1.        ]]
   
Translation =  [ 2.16298055 -1.42438431  1.54309862]
roll =  1.6593335679240577
pitch =  0.49472398572584053
yaw =  0.0662098822676542
```
We can verify that the results match the output from ROS. To simplify running different test cases, `test_joint_state_publisher.py` script is added to the project as a substitute for RViz joint state GUI.
```
# the first two elements are dummy values gripper left and right finger commands (not needed here)
$ python test_joint_state_publisher.py [0,0,-0.65,0.45,-0.36,0.95,0.79,0.49]

# In another terminal:
$ rosrun tf tf_echo base_link gripper_link
At time 1530840255.785
- Translation: [2.163, -1.424, 1.543]
- Rotation: in Quaternion [0.709, 0.189, -0.159, 0.660]
            in RPY (radian) [1.659, 0.495, 0.066]
            in RPY (degree) [95.073, 28.346, 3.794]
```

#### Inverse Kinematics
[ik]: ./misc/ik.png
![dh][ik]

To avoid unnecessary transformations and improve efficiency, **all the calculations are done in urdf reference frames**. Also, quaternions are used whenever possible (instead of Euler angles) for the same reasons and more accurate numerical evaluations.

To calculate the first three joint angles, the coordinates for link2 and link3 origins are projected to a plane passing through link2 and perpendicular to the horizontal plane (x'y') according to above diagram.

First, the coordinates of wrist center (WC) should be calculated. This can be done using the following equation:

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cphantom%7B%20%7D%5EB%20r_%7BWC%2FB_o%7D%20%3D%20%5Cphantom%7B%20%7D%5EB%20r_%7BG_o%2FB_o%7D%20-%20d_G%20.%20%5Cphantom%7B%20%7D%5EG_B%20R%5B%3A%2C3%5D%20)

That is, we should move back from the gripper location in the direction of third column of the rotation matrix from the base to gripper frame and with the length of d<sub>G</sub>. d<sub>G</sub> is distance of wrist center to the end-effector which can be found in the urdf file (second table in the previous section). We can rewrite the above equation in a more readable form of:

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%20wc_x%20%3D%20P_x%20-%20d_G%20.%20n_x%20)

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%20wc_y%20%3D%20P_y%20-%20d_G%20.%20n_y%20)

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%20wc_z%20%3D%20P_z%20-%20d_G%20.%20n_z%20)

where wc<sub>[x/y/z]</sub> are the coordinates of wrist center, P<sub>[x/y/z]</sub> are the coordinates of the gripper position (input to the IK problem) and n<sub>[x/y/z]</sub> are the elements of rotation matrix in the third column. Rotation matrix can be found by converting gripper pos quaternions (another input to the IK problem) to the matrix format.

Using WC coordinates, θ<sub>1</sub> can be calculated as

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Ctheta_1%20%3D%20%5Carctan2%28wc_y%2C%20wc_x%29%20)

From the above diagram, θ<sub>2</sub> and θ<sub>3</sub> can be found by the following equations:

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Ctheta_2%20%3D%20%5Cpi%20-%20a%20-%20%5Cgamma%20-%20%5Cpi%2F2%20)

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Ctheta_3%20%3D%20%5Cpi%20-%20b%20-%20%5Comega%20-%20%5Cpi%2F2%20)

γ is calculated using the projected coordinates of the wrist center and link2's origin:

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cgamma%20%3D%20%5Carctan2%28wc_%7By%27%7D%20-%20O_2_%7By%27%7D%2C%20wc_%7Bx%27%7D%20-%20O_2_%7Bx%27%7D%29%20)

ω is a fixed angle and can be calculated using the robot's urdf data and following equation:

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Comega%20%3D%20%5Carctan2%28-a_3%2C%20d_4%29%20)

angles a and b are calculated using law of cosines:

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20a%20%3D%20%5Carccos%20%5Cfrac%7BA%5E2-B%5E2-C%5E2%7D%7B2B.C%7D%20)

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20b%20%3D%20%5Carccos%20%5Cfrac%7BB%5E2-A%5E2-C%5E2%7D%7B2A.C%7D%20)

Using  θ<sub>1</sub>, θ<sub>2</sub> and θ<sub>3</sub>, we can find the rotation matrix from base to link3 and calculate the rotation matrix from link3 to the gripper using following equation:

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cphantom%7B%20%7D%5EG_3%20R%20%3D%20%5Ctext%7Binv%7D%5C%7B%7B%5Cphantom%7B%20%7D%5E3_B%20R%7D%5C%7D%20%5Cphantom%7B%20%7D%5EG_B%20R%20%3D%20%5Cphantom%7B%20%7D%5E3_B%20R%5ET%20%5Cphantom%7B%20%7D%5EG_B%20R%20%20)

Finally, θ<sub>4</sub>, θ<sub>5</sub> and  θ<sub>6</sub> are calculated as:

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Ctheta_4%20%3D%20%5Carctan2%28%5Cphantom%7B%20%7D%5EG_3%20R%5B2%2C%201%5D%2C%20-%5Cphantom%7B%20%7D%5EG_3%20R%5B3%2C%201%5D%29)

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Ctheta_5%20%3D%20%5Carctan2%28%5Csqrt%7B%5Cphantom%7B%20%7D%5EG_3%20R%5B2%2C%201%5D%5E%202%20%2B%20%5Cphantom%7B%20%7D%5EG_3%20R%5B3%2C%201%5D%20%5E%202%7D%2C%20%5Cphantom%7B%20%7D%5EG_3%20R%5B1%2C%201%5D%29)

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Ctheta_6%20%3D%20%5Carctan2%28%5Cphantom%7B%20%7D%5EG_3%20R%5B1%2C%202%5D%2C%20%5Cphantom%7B%20%7D%5EG_3%20R%5B1%2C%203%5D%29)


### Project Implementation

The IK equations described in the previous section are all implemented in `IK_server.py` and `IK_debug.py`. Since, the forward kinematic is not needed in `IK_server.py`, those codes are not included there. This makes the simulation run faster. Instead, FK codes are included in `kinematics.ipynb` notebook.

A preview of the simulation is included at the top of this page. Following picture is the screenshot of bin after task is completed for 9 different object locations

[comp]: ./output/completed.png
![comp]


