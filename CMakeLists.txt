CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
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

ADD_EXECUTABLE(YOLOv8 yolov8.h yolov8_onnx.h yolov8_seg.h yolov8_seg_onnx.h yolov8_utils.h
			 main.cpp yolov8.cpp yolov8_onnx.cpp yolov8_seg.cpp yolov8_seg_onnx.cpp yolov8_utils.cpp)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

TARGET_INCLUDE_DIRECTORIES(YOLOv8 PRIVATE "${ONNXRUNTIME_DIR}/include")

TARGET_COMPILE_FEATURES(YOLOv8 PRIVATE cxx_std_17)
TARGET_LINK_LIBRARIES(YOLOv8 ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ONNXRUNTIME_LIB})
