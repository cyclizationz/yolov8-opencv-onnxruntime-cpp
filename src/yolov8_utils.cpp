#pragma once
#include "yolov8_utils.h"
using namespace cv;
using namespace std;

bool CheckParams(int netHeight, int netWidth, const int *netStride,
                 int strideSize) {
  if (netHeight % netStride[strideSize - 1] != 0 ||
      netWidth % netStride[strideSize - 1] != 0) {
    cout << "Error:_netHeight and _netWidth must be multiple of max stride "
         << netStride[strideSize - 1] << "!" << endl;
    return false;
  }
  return true;
}

inline double GetTwoVectorsAngle(const cv::Point2d &v1, const cv::Point2d &v2) {
  double cosine_value = v1.dot(v2) / (cv::norm(v1) * cv::norm(v2));
  if (cosine_value > 1)
    cosine_value = 1;
  else if (cosine_value < -1)
    cosine_value = -1;

  double value = std::acos(cosine_value);
  return value;
}

inline bool IsThreePointsInLine(const cv::Point2d &p1, const cv::Point2d &p2,
                                const cv::Point2d &p3,
                                double radian_deviation) {
  cv::Point2d p1_p2 = p2 - p1;
  cv::Point2d p1_p3 = p3 - p1;
  // Check for overlapping points
  if (cv::norm(p1_p2) == 0 || cv::norm(p1_p3) == 0) {
    return true;
  }

  double radian_value_of_p1 = GetTwoVectorsAngle(p1_p2, p1_p3);

  cv::Point2d p3_p1 = p1 - p3;
  cv::Point2d p3_p2 = p2 - p3;
  // Check for overlapping points
  if (cv::norm(p3_p1) == 0 || cv::norm(p3_p2) == 0) {
    return true;
  }

  double radian_value_of_p3 = GetTwoVectorsAngle(p3_p1, p3_p2);

  if (radian_value_of_p1 <= radian_deviation &&
      radian_value_of_p3 <= radian_deviation) {
    return true;
  } else {
    return false;
  }
}

void LetterBox(const cv::Mat &image, cv::Mat &outImage, cv::Vec4d &params,
               const cv::Size &newShape, bool autoShape, bool scaleFill,
               bool scaleUp, int stride, const cv::Scalar &color) {
  if (false) {
    int maxLen = MAX(image.rows, image.cols);
    outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
    image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
    params[0] = 1;
    params[1] = 1;
    params[3] = 0;
    params[2] = 0;
  }

  cv::Size shape = image.size();
  float r = std::min((float)newShape.height / (float)shape.height,
                     (float)newShape.width / (float)shape.width);
  if (!scaleUp)
    r = std::min(r, 1.0f);

  float ratio[2]{r, r};
  int new_un_pad[2] = {(int)std::round((float)shape.width * r),
                       (int)std::round((float)shape.height * r)};

  auto dw = (float)(newShape.width - new_un_pad[0]);
  auto dh = (float)(newShape.height - new_un_pad[1]);

  if (autoShape) {
    dw = (float)((int)dw % stride);
    dh = (float)((int)dh % stride);
  } else if (scaleFill) {
    dw = 0.0f;
    dh = 0.0f;
    new_un_pad[0] = newShape.width;
    new_un_pad[1] = newShape.height;
    ratio[0] = (float)newShape.width / (float)shape.width;
    ratio[1] = (float)newShape.height / (float)shape.height;
  }

  dw /= 2.0f;
  dh /= 2.0f;

  if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1]) {
    cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
  } else {
    outImage = image.clone();
  }

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  params[0] = ratio[0];
  params[1] = ratio[1];
  params[2] = left;
  params[3] = top;
  cv::copyMakeBorder(outImage, outImage, top, bottom, left, right,
                     cv::BORDER_CONSTANT, color);
}

void GetMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos,
             std::vector<OutputSeg> &output, const MaskParams &maskParams) {
  // cout << maskProtos.size << endl;

  int seg_channels = maskParams.segChannels;
  int net_width = maskParams.netWidth;
  int seg_width = maskParams.segWidth;
  int net_height = maskParams.netHeight;
  int seg_height = maskParams.segHeight;
  float mask_threshold = maskParams.maskThreshold;
  Vec4f params = maskParams.params;
  Size src_img_shape = maskParams.srcImgShape;

  Mat protos = maskProtos.reshape(0, {seg_channels, seg_width * seg_height});

  Mat matmul_res = (maskProposals * protos).t();
  Mat masks = matmul_res.reshape(output.size(), {seg_width, seg_height});
  vector<Mat> maskChannels;
  split(masks, maskChannels);
  for (int i = 0; i < output.size(); ++i) {
    Mat dest, mask;
    // sigmoid
    cv::exp(-maskChannels[i], dest);
    dest = 1.0 / (1.0 + dest);

    Rect roi(int(params[2] / net_width * seg_width),
             int(params[3] / net_height * seg_height),
             int(seg_width - params[2] / 2), int(seg_height - params[3] / 2));
    dest = dest(roi);
    resize(dest, mask, src_img_shape, INTER_NEAREST);

    // crop
    Rect temp_rect = output[i].box;
    mask = mask(temp_rect) > mask_threshold;
    output[i].boxMask = mask;
  }
}

void GetMask2(const Mat &maskProposals, const Mat &mask_protos,
              OutputSeg &output, const MaskParams &maskParams) {
  //   cout << "Mask Protos: " << mask_protos << endl;

  int seg_channels = maskParams.segChannels;
  int net_width = maskParams.netWidth;
  int seg_width = maskParams.segWidth;
  int net_height = maskParams.netHeight;
  int seg_height = maskParams.segHeight;
  float mask_threshold = maskParams.maskThreshold;
  Vec4f params = maskParams.params;
  Size src_img_shape = maskParams.srcImgShape;

  // crop from mask_protos
  int rang_x =
      floor((output.box.x * params[0] + params[2]) / net_width * seg_width);
  int rang_y =
      floor((output.box.y * params[1] + params[3]) / net_height * seg_height);
  int rang_w =
      ceil(((output.box.x + output.box.width) * params[0] + params[2]) /
           net_width * seg_width) -
      rang_x;
  int rang_h =
      ceil(((output.box.y + output.box.height) * params[1] + params[3]) /
           net_height * seg_height) -
      rang_y;

  // If the following mask_protos(roi_rangs).clone() position reports an error,
  // it means that your output.box data is incorrect, or the rectangular box is
  // 1 pixel, uncomment the following section to prevent the error.
  rang_w = MAX(rang_w, 1);
  rang_h = MAX(rang_h, 1);
  if (rang_x + rang_w > seg_width) {
    if (seg_width - rang_x > 0)
      rang_w = seg_width - rang_x;
    else
      rang_x -= 1;
  }
  if (rang_y + rang_h > seg_height) {
    if (seg_height - rang_y > 0)
      rang_h = seg_height - rang_y;
    else
      rang_y -= 1;
  }

  vector<Range> roi_rangs;
  roi_rangs.push_back(Range(0, 1));
  roi_rangs.push_back(Range::all());
  roi_rangs.push_back(Range(rang_y, rang_h + rang_y));
  roi_rangs.push_back(Range(rang_x, rang_w + rang_x));

  // crop
  Mat temp_mask_protos = mask_protos(roi_rangs).clone();
  Mat protos = temp_mask_protos.reshape(0, {seg_channels, rang_w * rang_h});
  Mat matmul_res = (maskProposals * protos).t();
  Mat masks_feature = matmul_res.reshape(1, {rang_h, rang_w});
  Mat dest, mask;

  // sigmoid
  cv::exp(-masks_feature, dest);
  dest = 1.0 / (1.0 + dest);

  int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
  int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
  int width = ceil(net_width / seg_width * rang_w / params[0]);
  int height = ceil(net_height / seg_height * rang_h / params[1]);

  resize(dest, mask, Size(width, height), INTER_NEAREST);
  mask = mask(output.box - Point(left, top)) > mask_threshold;
  output.boxMask = mask;
  output.polygonPoints = binaryMaskToPolygon(mask, output.box);
}

std::vector<cv::Point> binaryMaskToPolygon(const cv::Mat &mask,
                                           const cv::Rect &box) {
  std::vector<Point> points;
  Point current_point;

  int left = box.x;
  int top = box.y;
  for (int row = 0; row < mask.rows; row++) {
    for (int col = 0; col < mask.cols; col++) {
      if (mask.at<uchar>(row, col) != 0) {
        current_point = {col, row};
        if (points.size() <= 2 ||
            (!IsThreePointsInLine(points[points.size() - 3],
                                  points[points.size() - 2], current_point,
                                  0.017) &&
             points[points.size() - 2].y != current_point.y)) {

          points.push_back({col, row});
        }
      }
    }
  }
  for (auto &point : points) {
    point.x += left;
    point.y += top;
  }
  return points;
}

void DrawPred(Mat &img, vector<OutputSeg> result,
              std::vector<std::string> classNames, vector<Scalar> color,
              const std::string &out_path) {
  Mat mask = img.clone();
  for (int i = 0; i < result.size(); i++) {
    int left, top;
    left = result[i].box.x;
    top = result[i].box.y;
    int color_num = i;
    rectangle(img, result[i].box, color[result[i].id], 2, 8);
    if (result[i].boxMask.rows && result[i].boxMask.cols > 0)
      mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
    string label =
        classNames[result[i].id] + ":" + to_string(result[i].confidence);
    // cout << label << endl;
    int baseLine;
    Size labelSize =
        getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    // rectangle(frame, Point(left, top - int(1.5 * labelSize.height)),
    // Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0,
    // 255, 0), FILLED);
    putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1,
            color[result[i].id], 2);
  }
  addWeighted(img, 0.5, mask, 0.5, 0, img); // add mask to src
  // imshow(img);
  imwrite(out_path, img);
  // waitKey();
  // destroyAllWindows();
}

void check_opencv() {
  // Version check code is from Satya Mallick's  post
  // https://learnopencv.com/how-to-find-opencv-version-python-cpp/
  std::cout << "OpenCV version : " << CV_VERSION << std::endl;
  std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
  std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
  std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;
  std::cout << "*****************************************************"
            << std::endl;

  // Test of GPU availability and functions
  int cudaEnabDevCount = cv::cuda::getCudaEnabledDeviceCount();

  if (cudaEnabDevCount)
    std::cout << "Number of available CUDA device(s): " << cudaEnabDevCount
              << std::endl;
  else
    std::cout << "You don't have any available CUDA device(s)" << std::endl;
  std::cout << "*****************************************************"
            << std::endl;

  std::cout << "List of all available CUDA device(s):" << std::endl;
  for (int devId = 0; devId < cudaEnabDevCount; ++devId) {
    cv::cuda::setDevice(devId);
    std::cout << "Available ";
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
  }
  std::cout << "*****************************************************"
            << std::endl;

  cv::cuda::DeviceInfo cudaDeviceInfo;
  bool devCompatib = false;

  std::cout << "List of all compatiable CUDA device(s):" << std::endl;
  for (int devId = 0; devId < cudaEnabDevCount; ++devId) {
    cudaDeviceInfo = cv::cuda::DeviceInfo(devId);
    devCompatib = cudaDeviceInfo.isCompatible();

    if (devCompatib)
      std::cout << "Compatiable ";
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
  }
  std::cout << "*****************************************************"
            << std::endl;
}