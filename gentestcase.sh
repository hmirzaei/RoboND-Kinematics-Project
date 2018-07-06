echo "[["
rosrun tf tf_echo base_link gripper_link -n 2> /dev/null | grep Trans | sed 's/.*.\(\[.*\]\)/\1/g'
echo ","
rosrun tf tf_echo base_link gripper_link -n 2> /dev/null | grep Quat | sed 's/.*.\(\[.*\]\)/\1/g'
echo "],"
rosrun tf tf_echo base_link link_5 -n 2> /dev/null | grep Trans | sed 's/.*.\(\[.*\]\)/\1/g'
echo ","
rostopic echo -n1 /joint_states  | grep pos | perl -pe 's|.*?\[.*?, .*?, (.*\]).*|\[\1|g'
echo "]"
