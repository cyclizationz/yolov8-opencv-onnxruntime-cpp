#include <assert.h>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolov8.h"
#include "yolov8_onnx.h"
#include "yolov8_seg.h"
#include "yolov8_seg_onnx.h"
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;
using namespace dnn;
namespace fs = std::filesystem;

// number of classes
const int NUM_CLS = 80;

// generate random mask color
static vector<Scalar> color;
static void generate_mask_color() {
  srand(time(0));
  for (int i = 0; i < NUM_CLS; i++) {
    int b = rand() % 256;
    int g = rand() % 256;
    int r = rand() % 256;
    color.push_back(Scalar(b, g, r));
  }
}

/**
\TODO: modify ReadModel to read only once per inference.
*/
template <typename _Tp>
int yolov8(_Tp &cls, Mat &img,const string &model_path,
           const std::string &out_path) {

  Net net;
  if (cls.ReadModel(net, model_path, false)) {
    cout << "read net ok!" << endl;
  } else {
    return -1;
  }
  vector<OutputSeg> result;

  if (cls.Detect(img, net, result)) {
    DrawPred(img, result, cls._className, color, out_path);
  } else {
    cout << "Detect Failed!" << endl;
  }
  system("pause");
  return 0;
}

template <typename _Tp>
int yolov8_onnx(_Tp &cls, Mat &img,const string &model_path,
                const std::string &out_path) {

  if (cls.ReadModel(model_path, false)) {
    cout << "read net ok!" << endl;
  } else {
    return -1;
  }
  vector<OutputSeg> result;
  if (cls.OnnxDetect(img, result)) {
    DrawPred(img, result, cls._className, color, out_path);
  } else {
    cout << "Detect Failed!" << endl;
  }
  system("pause");
  return 0;
}

/**
 * \TODO: wrap yolo with process_image and process_image_onnx to parse read
 * params(isCuda, cudaID, warmUp)
 */
void process_image(const std::string &img_path, const std::string &model_path,
                   const std::string &task, const std::string &out_path) {
  auto entry = fs::directory_entry(img_path);
  assert(entry.is_regular_file() && (entry.path().extension() == ".jpg" ||
                                     entry.path().extension() == ".JPG" ||
                                     entry.path().extension() == ".jpeg" ||
                                     entry.path().extension() == ".JPEG" ||
                                     entry.path().extension() == ".png" ||
                                     entry.path().extension() == ".PNG"));
  cv::Mat img = cv::imread(img_path);
  if (img.empty()) {
    std::cout << "Error: Failed to read image " << img_path << std::endl;
    return;
  }

  // Print progress information
  std::cout << "Processed image: " << img_path << std::endl;
  if (task == "detect") {
    Yolov8 task_detect;
    yolov8(task_detect, img, model_path, out_path);
  } else if (task == "segment") {
    Yolov8Seg task_segment;
    yolov8(task_segment, img, model_path, out_path);
  }
}

int main(int argc, char *argv[]) {

  std::string input_dir = "../images/";
  std::string task = "segment";
  std::string model_path = "../models/yolov8n-seg-FC5-sim.onnx";
  std::string output_dir = "../outputs/";
  bool enable_onnxruntime = false;
  bool isCuda = false;
  int cudaID = 0;
  bool warmUp = true;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--task=segment" || arg == "--task=detect") {
      task = arg.substr(arg.find('=') + 1);
    } else if (arg == "--onnx") {
      enable_onnxruntime = true;
    } else if (arg.find("--cuda:") != std::string::npos) {
      isCuda = true;
      cudaID = std::stoi(arg.substr(arg.find(':') + 1));
    } else if (arg.find("-i:") != std::string::npos) {
      input_dir = arg.substr(arg.find(':') + 1, arg.length() - 3);
    } else if (arg.find("-o:") != std::string::npos) {
      output_dir = arg.substr(arg.find(':') + 1, arg.length() - 3);
    }
  }
  // Mat img = imread(img_path);

  // Yolov8 task_detect;
  // Yolov8Seg task_segment;
  // Yolov8Onnx task_detect_onnx;
  // Yolov8SegOnnx task_segment_onnx;

  // yolov8(task_detect,img,detect_model_path,output_dir);    //Opencv detect
  // yolov8(task_segment,img,seg_model_path,output_dir);   //opencv segment
  // yolov8_onnx(task_detect_onnx,img,detect_model_path,output_dir);
  // //onnxruntime detect
  // yolov8_onnx(task_segment_onnx,img,seg_model_path,output_dir); //onnxruntime
  // segment

  if (input_dir.empty()) {
    std::cout << "Error: No input path or directories specified." << std::endl;
    return 1;
  }

  if (fs::is_directory(input_dir)) {
    // Process all images in the directory
    for (const auto &entry : fs::directory_iterator(input_dir)) {
    }
  } else {
    if (!fs::exists(input_dir)) {
      std::cout << "Error: Input path: " << input_dir << " not found."
                << std::endl;
      return 1;
    }
    // Process single image file
  }

  return 0;
}