#include <assert.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// #include "yolov8.h"
#include "yolov8_onnx.h"
// #include "yolov8_seg.h"
#include "yolov8_seg_onnx.h"
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;
// using namespace dnn;
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

/*
template <typename _Tp>
int yolov8(_Tp &cls, Mat &img, const string &model_path,
           const std::string &out_path, bool isCuda = false) {
  Net net;
  string out_image_path = out_path + ".png";
  string out_json_path = out_path + ".json";
  // cout << out_image_path << " " << out_json_path << endl;
  if (cls.ReadModel(net, model_path, isCuda)) {
    vector<OutputSeg> result;

    if (cls.Detect(img, net, result)) {
      ofstream result_stream;
      result_stream.open(out_json_path);
      if (result_stream.is_open()) {
        result_stream << "{\n";
        result_stream << "  \"version\": \"5.2.1\",\n";
        result_stream << "  \"flags\": {},\n";
        result_stream << "  \"shapes\": [\n";

        for (size_t i = 0; i < result.size(); i++) {
          cout << "Instance detected/segmented: " << result[i].id
               << " bbox: " << result[i].box
               << " ,confidence: " << result[i].confidence << endl;

          result_stream << "    {\n";
          result_stream << "      \"label\": \"gun\",\n";
          result_stream << "      \"points\": [\n";

          if (std::is_same<_Tp, Yolov8>::value) {
            // If the task is detection
            result_stream << "        [\n";
            result_stream << "          " << result[i].box.x << ",\n "
                          << result[i].box.y << "\n";
            result_stream << "        ],\n";
            result_stream << "        [\n";
            result_stream << "          "
                          << result[i].box.x + result[i].box.width
                          << ",\n          "
                          << result[i].box.y + result[i].box.height << "\n";
            result_stream << "        ]\n";
          } else if (std::is_same<_Tp, Yolov8Seg>::value) {
            // If the task is segmentation
            for (size_t j = 0; j < result[i].polygonPoints.size(); j++) {
              result_stream << "        [\n";
              result_stream << "          " << result[i].polygonPoints[j].x
                            << ",\n          " << result[i].polygonPoints[j].y
                            << "\n";
              result_stream << "        ]";
              if (j != result[i].polygonPoints.size() - 1) {
                result_stream << ",\n";
              } else {
                result_stream << "\n";
              }
            }
          }

          result_stream << "      ],\n";
          result_stream << "      \"group_id\": null,\n";
          result_stream << "      \"description\": \"\",\n";
          result_stream << "      \"shape_type\": \"polygon\",\n";
          result_stream << "      \"flags\": {}\n";
          result_stream << "    }";

          if (i != result.size() - 1) {
            result_stream << ",";
          }
          result_stream << "\n";
        }

        result_stream << "  ]\n";
        result_stream << "}\n";

      } else {
        cout << "Result .json files" << out_json_path << "cannot be opened"
             << endl;
      }
      DrawPred(img, result, cls._className, color, out_image_path);
      result_stream.close();
    } else {
      cout << "Detect Failed!" << endl;
    }
    return 0;
  } else {
    return -1;
  }
}
*/


template <typename _Tp>
int yolov8_onnx(_Tp &cls, Mat &img, const string &model_path,
                const std::string &out_path, bool isCuda = false,
                int cudaID = 0, bool warmUp = true) {

  // cout << "====>this is yolov8_onnx() in main.cpp, line131" << endl;

  string out_image_path = out_path + ".png";
  string out_json_path = out_path + ".json";
  if (cls.modelInitialized() ||
      cls.ReadModel(model_path, isCuda, cudaID, warmUp)) {
    vector<OutputSeg> result;
    if (cls.OnnxDetect(img, result)) {
      ofstream result_stream;
      result_stream.open(out_json_path);
      if (result_stream.is_open()) {
        result_stream << "{\n";
        result_stream << "  \"version\": \"5.2.1\",\n";
        result_stream << "  \"flags\": {},\n";
        result_stream << "  \"shapes\": [\n";

        for (size_t i = 0; i < result.size(); i++) {
          cout << "Instance detected/segmented: " << result[i].id
               << " bbox: " << result[i].box
               << " ,confidence: " << result[i].confidence << endl;

          result_stream << "    {\n";
          result_stream << "      \"label\": \"gun\",\n";
          result_stream << "      \"points\": [\n";

          if (std::is_same<_Tp, Yolov8Onnx>::value) {
            // If the task is detection
            result_stream << "        [\n";
            result_stream << "          " << result[i].box.x << ", "
                          << result[i].box.y << "\n";
            result_stream << "        ],\n";
            result_stream << "        [\n";
            result_stream << "          "
                          << result[i].box.x + result[i].box.width
                          << ",\n          "
                          << result[i].box.y + result[i].box.height << "\n";
            result_stream << "        ]\n";
          } else if (std::is_same<_Tp, Yolov8SegOnnx>::value) {
            // If the task is segmentation
            for (size_t j = 0; j < result[i].polygonPoints.size(); j++) {
              result_stream << "        [\n";
              result_stream << "          " << result[i].polygonPoints[j].x
                            << ",\n          " << result[i].polygonPoints[j].y
                            << "\n";
              result_stream << "        ]";
              if (j != result[i].polygonPoints.size() - 1) {
                result_stream << ",\n";
              } else {
                result_stream << "\n";
              }
            }
          }

          result_stream << "      ],\n";
          result_stream << "      \"group_id\": null,\n";
          result_stream << "      \"description\": \"\",\n";
          result_stream << "      \"shape_type\": \"polygon\",\n";
          result_stream << "      \"flags\": {}\n";
          result_stream << "    }";

          if (i != result.size() - 1) {
            result_stream << ",";
          }
          result_stream << "\n";
        }

        result_stream << "  ]\n";
        result_stream << "}\n";

      } else {
        cout << "Result .json files" << out_json_path << "cannot be opened"
             << endl;
      }

      DrawPred(img, result, cls._className, color, out_image_path);
      result_stream.close();
    } else {
      cout << "Detect Failed!" << endl;
    }
    return 0;
  } else {
    return -1;
  }
}

/**
 * \TODO: wrap yolo with process_image and process_image_onnx to parse read
 * params(isCuda, cudaID, warmUp)
 */
/*
template <typename _Tp>
void process_image(_Tp &cls, const std::string &img_path,
                   const std::string &model_path, const std::string &out_path,
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
  yolov8(cls, img, model_path, out_path, isCuda);
}
*/

template <typename _Tp>
void process_image_onnx(_Tp &cls, const std::string &img_path,
                        const std::string &model_path,
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
  // yolov8_onnx(task_detect_onnx,img,detect_model_path,output_dir);
  // //onnxruntime detect
  // yolov8_onnx(task_segment_onnx,img,seg_model_path,output_dir); //onnxruntime
  // segment
  yolov8_onnx(cls, img, model_path, out_path, isCuda, cudaID, warmUp);
}

int main(int argc, char *argv[]) {

  std::string input_dir = "../images/";
  std::string task = "segment";
  std::string model_path = "../models/yolov8n-seg-FC5-sim.onnx";
  std::string output_dir = "../outputs/";
  bool enable_onnxruntime = true;
  bool isCuda = false;
  int cudaID = 0;
  bool warmUp = true;
  generate_mask_color();
  // check_opencv();

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
    } else if (arg.find("-m:") != std::string::npos) {
      model_path = arg.substr(arg.find(':') + 1, arg.length() - 3);
    }
  }
  // Mat img = imread(img_path);

  // Yolov8 task_detect;
  // Yolov8Seg task_segment;
  Yolov8Onnx task_detect_onnx;
  Yolov8SegOnnx task_segment_onnx;

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
        if (task == "detect")
          process_image_onnx(task_detect_onnx, entry.path().c_str(), model_path,
                             out_path, isCuda, cudaID, warmUp);
        else
          process_image_onnx(task_segment_onnx, entry.path().c_str(),
                             model_path, out_path, isCuda, cudaID, warmUp);
        std::cout << "Prediction saved to:" << out_path << ".png" << std::endl;
        std::cout << "Result .json files saved to:" << out_path << ".json"
                  << std::endl;
        std::cout << "==================================================="
                  << std::endl;
      } 
      // else {
      //   if (task == "detect")
      //     process_image(task_detect, entry.path().c_str(), model_path, out_path,
      //                   isCuda);
      //   else
      //     process_image(task_segment, entry.path().c_str(), model_path,
      //                   out_path, isCuda);
      //   std::cout << "Prediction saved to:" << out_path << ".png" << std::endl;
      //   std::cout << "Result .json files saved to:" << out_path << ".json"
      //             << std::endl;
      //   std::cout << "==================================================="
      //             << std::endl;
      // }
    }
  }
  // else if (fs::is_regular_file(input_dir)) { // Process single image file
  //   std::string out_path = output_dir + fs::path(input_dir).stem().string();
  //   if (enable_onnxruntime) {
  //     process_image_onnx(input_dir, model_path, task, out_path, isCuda,
  //     cudaID,
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