cmake_minimum_required(VERSION 3.0.2)
project(box_mot)

find_package(
  catkin REQUIRED
  COMPONENTS sensor_msgs
             cv_bridge
             rospy
             std_msgs
             message_generation
)

add_message_files(FILES Features.msg)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES yolact_ros
  CATKIN_DEPENDS
    sensor_msgs
    cv_bridge
    rospy
    message_runtime
  #  DEPENDS system_lib
)

install(
  PROGRAMS scripts/completion.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
