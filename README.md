# yolov8-opencv-onnxruntime-cpp
## Use OpenCV-dnn and ONNXRuntime to deploy yolov8 detection and instance segmentation model<br>
Based upon:
yolov8:https://github.com/ultralytics/ultralytics

## requirements for opencv-dnn
1. > OpenCV>=4.7.0<br>
OpenCV>=4.7.0<br>
OpenCV>=4.7.0<br>

2. export for opencv-dnn:</br>
> ```yolo export model=path/to/model.pt format=onnx dynamic=False  opset=12```</br>

## requirements for onnxruntime （only yolo*_onnx.h/cpp）
>opencv>=4.5.0 </br>
ONNXRuntime>=1.9.0 </br>

## Build OpenCV:

### Install minimal prerequisites (Ubuntu 18.04 as reference)
>sudo apt update && sudo apt install -y </br>
cmake g++ wget unzip </br>
### Download and unpack sources
>wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip </br>
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip</br>
unzip opencv.zip</br>
unzip opencv_contrib.zip</br>
### Create build directory and switch into it
>mkdir -p build && cd build
### Configure
>cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
### Build
> cmake --build .
### Specify OpenCV_DIR in CMakeLists.txt
>SET (OpenCV_DIR /path/to/OpenCV/Build)

## Build ONNXRuntime
Checkout the source tree:
> git clone --recursive https://github.com/Microsoft/onnxruntime.git
 cd onnxruntime

Install CMake-3.26 or higher. </br>
On Windows please run:
> python -m pip install cmake </br>
where cmake </br>

On Linux please run:
> python3 -m pip install cmake </br>
which cmake </br>

### Build Instructions </br>
#### WINDOWS
Open Developer Command Prompt for Visual Studio version you are going to use. This will properly setup the environment including paths to your compiler, linker, utilities and header files.

>.\build.bat --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync

The default Windows CMake Generator is Visual Studio 2022. For Visual Studio 2019 add --cmake_generator "Visual Studio 16 2019".

We recommend using Visual Studio 2022.

If you want to build an ARM64 binary on a Windows ARM64 machine, you can use the same command above. Just be sure that your Visual Studio, CMake and Python are all ARM64 version.

If you want to cross-compile an ARM32 or ARM64 or ARM64EC binary on a Windows x86 machine, you need to add “–arm” or “–arm64” or “–arm64ec” to the build command above.

When building on x86 Windows without “–arm” or “–arm64” or “–arm64ec” args, the built binaries will be 64-bit if your python is 64-bit, or 32-bit if your python is 32-bit,

#### LINUX
> ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync

### Specify ONNXRuntime_LIB in CMakeLists.txt
ONNX do not provide **find_packages** in CMake, so we should configure cmake using the following script.
```cmake
SET(ONNXRUNTIME_ROOT_PATH /root/autodl-tmp/onnxruntime)
SET(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/)
SET(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)

INCLUDE_DIRECTORIES(${ONNXRUNTIME_INCLUDE_DIRS})
```

## Build project and run

To run this inference in terminal, first change classes list in every .h header files to your customized dataset.</br>
Then **uncomment** the model you want to use in the main function.

>mkdir -p build</br>
cmake . </br>
cd build </br>
cmake --build .. </br>

Then you can specify parameters in command line like:
>YOLOv8 -i:./images --task=segment --onnx --cuda:0

## TODO:
1. Modify multi-file inference, to read models only once per inference.
2. Add .yaml interpreter to use Python configs directly.