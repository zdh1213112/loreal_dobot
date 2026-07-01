cmake_minimum_required(VERSION 3.8)
project(barcode_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_library(ZBAR_LIBRARY zbar)

# ========================================================
# 💡 引入 ONNXRuntime
set(ONNXRUNTIME_DIR "/home/zdh/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)
# ========================================================

set(NODE_LIST 
  barcode_node 
  yolo_barcode_node
  d405_yolo_barcode_node
  d405_yolo_barcode_node1
  d405_yolo_barcode_node2
  d405_yolo_barcode_node3
  d405_yolo_barcode_node3_onnx_flex
  d405_yolo_barcode_node3_onnx_flex_filtered
  d405_yolo_barcode_node4
)

foreach(node ${NODE_LIST})
  add_executable(${node} src/${node}.cpp)
  
  target_include_directories(${node} PUBLIC ${OpenCV_INCLUDE_DIRS} ${ZBAR_INCLUDE_DIRS})
  
  # 💡 注意这里加上了 onnxruntime
  target_link_libraries(${node} ${OpenCV_LIBS} realsense2::realsense2 zbar onnxruntime)
  
  ament_target_dependencies(${node} rclcpp std_msgs)
  install(TARGETS ${node} DESTINATION lib/${PROJECT_NAME})
endforeach()

install(PROGRAMS
  scripts/d405_yolo_barcode_node3_pt_flex.py
  scripts/d405_yolo_barcode_node3_pt_flex_filtered.py
  scripts/d405_yolo_barcode_node3_pt_flex_lowglare_panel.py
  scripts/d405_yolo_pt_common.py
  scripts/d405_yolo_pt_common_lowglare_panel.py
  scripts/vision_panel_viewer.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
