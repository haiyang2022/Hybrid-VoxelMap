cmake_minimum_required(VERSION 3.8)
project(hybrid_voxel_map)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-O3 -pthread -fexceptions)
add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
include(ProcessorCount)
ProcessorCount(N)
if(N EQUAL 0)
  set(N 1)
endif()
message(STATUS "Processor number: ${N}")

set(HVM_OMP_NUM_THREADS 0 CACHE STRING "OpenMP thread count for Hybrid-VoxelMap. 0 chooses a conservative default.")
if(HVM_OMP_NUM_THREADS GREATER 0)
  set(PROC_NUM ${HVM_OMP_NUM_THREADS})
elseif(N GREATER 8)
  set(PROC_NUM 8)
else()
  set(PROC_NUM ${N})
endif()

if(PROC_NUM GREATER 1)
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()
message(STATUS "OpenMP residual matching threads: ${PROC_NUM}")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP QUIET)
find_package(livox_ros_driver2 QUIET)

if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
if(OpenMP_C_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

if(livox_ros_driver2_FOUND)
  add_definitions(-DHAVE_LIVOX_ROS_DRIVER2)
  message(STATUS "livox_ros_driver2 found: Livox CustomMsg input enabled")
else()
  message(WARNING "livox_ros_driver2 not found: lidar_type=1 (Livox Avia CustomMsg) will be disabled")
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Pose6D.msg"
  "msg/States.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(voxel_mapping_odom
  src/voxelMapping.cpp
  src/preprocess.cpp
)
add_executable(kitti_sequence_publisher_cpp
  src/kitti_sequence_publisher.cpp
)
add_executable(kitti_gt_semantic_bridge
  src/kitti_gt_semantic_bridge.cpp
)
if(livox_ros_driver2_FOUND)
  add_executable(mcd_mid70_semantic_gt_bridge
    src/mcd_mid70_semantic_gt_bridge.cpp
  )
  add_executable(livox_custom_to_pointcloud2
    src/livox_custom_to_pointcloud2.cpp
  )
endif()

ament_target_dependencies(voxel_mapping_odom
  rclcpp
  builtin_interfaces
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  pcl_conversions
  tf2
  tf2_geometry_msgs
  tf2_ros
)
if(livox_ros_driver2_FOUND)
  ament_target_dependencies(voxel_mapping_odom livox_ros_driver2)
endif()
ament_target_dependencies(kitti_sequence_publisher_cpp
  rclcpp
  sensor_msgs
)
ament_target_dependencies(kitti_gt_semantic_bridge
  rclcpp
  sensor_msgs
)
if(livox_ros_driver2_FOUND)
  ament_target_dependencies(mcd_mid70_semantic_gt_bridge
    rclcpp
    sensor_msgs
    livox_ros_driver2
  )
  target_link_libraries(mcd_mid70_semantic_gt_bridge ${PCL_LIBRARIES})
  ament_target_dependencies(livox_custom_to_pointcloud2
    rclcpp
    sensor_msgs
    livox_ros_driver2
  )
endif()

target_link_libraries(voxel_mapping_odom ${PCL_LIBRARIES})
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(voxel_mapping_odom "${cpp_typesupport_target}")

install(TARGETS voxel_mapping_odom kitti_sequence_publisher_cpp kitti_gt_semantic_bridge
  DESTINATION lib/${PROJECT_NAME}
)
if(livox_ros_driver2_FOUND)
  install(TARGETS mcd_mid70_semantic_gt_bridge livox_custom_to_pointcloud2
    DESTINATION lib/${PROJECT_NAME}
  )
endif()
install(PROGRAMS
  scripts/kitti_sequence_publisher.py
  scripts/hilti2022_cloud_bridge.py
  scripts/hilti2022_playback_ready.py
  scripts/run_hilti2022.py
  scripts/generic_semantic_field_bridge.py
  scripts/salsanext_semantic_ros2_node.py
  scripts/salsanext_mcd_livox_semantic_ros2_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config launch rviz_cfg
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
