#!/usr/bin/env python

import roslib
import rospy
import sys
import ast

from sensor_msgs.msg import JointState as JointStatePR2


class JointStateMessage():
    def __init__(self, name, position, velocity, effort):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.effort = effort

class JointStatePublisher():
    def __init__(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        
        rate = rospy.get_param('~rate', 5)
        r = rospy.Rate(rate)
        
        namespace = rospy.get_namespace()
        self.joints = rospy.get_param(namespace + '/joints', '')
                                                                
        self.joint_states = dict({})
        
        # Start publisher
        self.joint_states_pub = rospy.Publisher('/joint_states', JointStatePR2)
       
        rospy.loginfo("Starting Joint State Publisher at " + str(rate) + "Hz")
       
        while not rospy.is_shutdown():
            self.publish_joint_states()
            r.sleep()
                  
    def publish_joint_states(self):
        # Construct message & publish joint states
        msg = JointStatePR2()
        msg.name =  ['right_gripper_finger_joint', 'left_gripper_finger_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        msg.position = ast.literal_eval(sys.argv[1])
        msg.velocity = []
        msg.effort = []
       
        for joint in self.joint_states.values():
            msg.name.append(joint.name)
            msg.position.append(joint.position)
            msg.velocity.append(joint.velocity)
            msg.effort.append(joint.effort)
           
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'base_link'
        self.joint_states_pub.publish(msg)
        
if __name__ == '__main__':
    try:
        s = JointStatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException: pass

