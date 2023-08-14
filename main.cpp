#include <assert.h>
#include <filesystem>
#include <fstream>
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
int yolov8(_Tp &cls, Mat &img, const string &model_path,
           const std::string &out_path, bool isCuda = false) {

  Net net;
  string out_image_path = out_path + ".png";
  string out_txt_path = out_path + ".txt";
  // cout << out_image_path << " " << out_txt_path << endl;
  if (cls.ReadModel(net, model_path, isCuda)) {
    cout << "read net ok!" << endl;
  } else {
    return -1;
  }
  vector<OutputSeg> result;

  if (cls.Detect(img, net, result)) {
    ofstream result_stream;
    result_stream.open(out_txt_path);
    if (result_stream.is_open()) {
      for (auto seg : result) {
        cout << "Instance detected/segmented: " << seg.id
             << " ,confidence: " << seg.confidence << endl;
        result_stream << seg.id << std::endl;
        result_stream << seg.box << std::endl;
        result_stream << seg.boxMask << std::endl;
        result_stream << seg.confidence << std::endl;
        cout << "Result text files saved to:" << out_txt_path << endl;
      }
    } else {
      cout << "Result text files" << out_txt_path << "cannot be opened" << endl;
    }
    DrawPred(img, result, cls._className, color, out_image_path);
    result_stream.close();
  } else {
    cout << "Detect Failed!" << endl;
  }
  return 0;
}

template <typename _Tp>
int yolov8_onnx(_Tp &cls, Mat &img, const string &model_path,
                const std::string &out_path, bool isCuda = false,
                int cudaID = 0, bool warmUp = true) {

  string out_image_path = out_path + ".png";
  string out_txt_path = out_path + ".txt";
  // cout << out_image_path << " " << out_txt_path << endl;

  if (cls.ReadModel(model_path, isCuda, cudaID, warmUp)) {
    cout << "read net ok!" << endl;
  } else {
    return -1;
  }

  vector<OutputSeg> result;
  if (cls.OnnxDetect(img, result)) {
    ofstream result_stream;
    result_stream.open(out_txt_path);
    if (result_stream.is_open()) {
      for (auto seg : result) {
        cout << "Instance detected/segmented: " << seg.id
             << " ,confidence: " << seg.confidence << endl;
        result_stream << seg.id << std::endl;
        result_stream << seg.box << std::endl;
        result_stream << seg.boxMask << std::endl;
        result_stream << seg.confidence << std::endl;
        cout << "Result text files saved to:" << out_txt_path << endl;
      }
    } else {
      cout << "Result text files" << out_txt_path << "cannot be opened" << endl;
    }
    DrawPred(img, result, cls._className, color, out_image_path);
    result_stream.close();
  } else {
    cout << "Detect Failed!" << endl;
  }
  return 0;
}

/**
 * \TODO: wrap yolo with process_image and process_image_onnx to parse read
 * params(isCuda, cudaID, warmUp)
 */
void process_image(const std::string &img_path, const std::string &model_path,
                   const std::string &task, const std::string &out_path,
                   bool isCuda) {
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
    yolov8(task_detect, img, model_path, out_path, isCuda);
  } else if (task == "segment") {
    Yolov8Seg task_segment;
    yolov8(task_segment, img, model_path, out_path, isCuda);
  }
}

void process_image_onnx(const std::string &img_path,
                        const std::string &model_path, const std::string &task,
                        const std::string &out_path, bool isCuda, int cudaID,
                        bool warmUp) {
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
  // yolov8_onnx(task_detect_onnx,img,detect_model_path,output_dir); //onnxruntime detect
  // yolov8_onnx(task_segment_onnx,img,seg_model_path,output_dir); //onnxruntime segment
  if (task == "detect") {
    Yolov8Onnx task_detect;
    yolov8_onnx(task_detect, img, model_path, out_path, isCuda, cudaID, warmUp);
  } else if (task == "segment") {
    Yolov8SegOnnx task_segment;
    yolov8_onnx(task_segment, img, model_path, out_path, isCuda, cudaID,
                warmUp);
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
  generate_mask_color();

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--task=segment" || arg == "--task=detect") {
      task = arg.substr(arg.find('=') + 1);
    } else if (arg == "--onnx") {
      enable_onnxruntime = true;
    } else if (arg == "--cuda") {
      isCuda = true;
    } else if (arg.find("--cuda:") != std::string::npos) {
      isCuda = true;
      cudaID = std::stoi(arg.substr(arg.find(':') + 1));
    } else if (arg == "--disable-warmup") {
      warmUp == false;
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


  if (input_dir.empty()) {
    std::cout << "Error: No input path or directories specified." << std::endl;
    return 1;
  }
  if (output_dir.empty()) {
    std::cout << "Error: No input path or directories specified." << std::endl;
    return 1;
  }
  if (!fs::exists(output_dir)) {
    fs::create_directory(output_dir);
  }

  if (fs::is_directory(input_dir)) {
    // Process all images in the directory
    for (const auto &entry : fs::directory_iterator(input_dir)) {
      std::string out_path = output_dir + entry.path().stem().string();
      if (enable_onnxruntime) {
        process_image_onnx(entry.path().c_str(), model_path, task, out_path,
                           isCuda, cudaID, warmUp);
        std::cout << "Prediction saved to:" << out_path << std::endl;
      } else {
        process_image(entry.path().c_str(), model_path, task, out_path, isCuda);
        std::cout << "Prediction saved to:" << out_path << std::endl;
      }
    }
  } 
  // else if (fs::is_regular_file(input_dir)) { // Process single image file
  //   std::string out_path = output_dir + fs::path(input_dir).stem().string();
  //   if (enable_onnxruntime) {
  //     process_image_onnx(input_dir, model_path, task, out_path, isCuda, cudaID,
  //                        warmUp);
  //     std::cout << "Prediction saved to:" << out_path << std::endl;
  //   } else {
  //     process_image(input_dir, model_path, task, out_path, isCuda);
  //     std::cout << "Prediction saved to:" << out_path << std::endl;
  //   }
  // } 
  else {
    if (!fs::exists(input_dir)) {
      std::cout << "Error: Input path: " << input_dir << " not found."
                << std::endl;
      return 1;
    }
  }

  return 0;
}