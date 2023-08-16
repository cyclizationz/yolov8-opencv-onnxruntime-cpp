cmake_minimum_required(VERSION 3.26)
project(YOLOv8)

#opencv root
SET (OpenCV_DIR /root/autodl-tmp/build)  
#onnxruntime
SET(ONNXRUNTIME_ROOT_PATH /root/autodl-tmp/onnxruntime)
SET(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/)
SET(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${ONNXRUNTIME_INCLUDE_DIRS})

AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src  SRC_LIST)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)

ADD_EXECUTABLE(YOLOv8 
			 main.cpp ./src/yolov8.cpp ./src/yolov8_onnx.cpp ./src/yolov8_seg.cpp ./src/yolov8_seg_onnx.cpp ./src/yolov8_utils.cpp)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

TARGET_INCLUDE_DIRECTORIES(YOLOv8 PRIVATE "${ONNXRUNTIME_DIR}/include")

TARGET_COMPILE_FEATURES(YOLOv8 PRIVATE cxx_std_17)
TARGET_LINK_LIBRARIES(YOLOv8 ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ONNXRUNTIME_LIB})
