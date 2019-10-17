cmake_minimum_required(VERSION 2.8.3)
project(resnet)




find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation 
  geometry_msgs
  
)


add_message_files(
FILES
barrel.msg
barrels.msg
box2d.msg
box2ds.msg
BoundingBox.msg
BoundingBoxArray.msg
Object.msg
ObjectArray.msg
)

generate_messages(
DEPENDENCIES
std_msgs
sensor_msgs
geometry_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime sensor_msgs )



include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

