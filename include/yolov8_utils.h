#pragma once
#include<iostream>
#include <vector>
#include <numeric>
#include<opencv2/opencv.hpp>

#define YOLO_P6 false 
#define _CV_DNN_ENABLED false
#define ORT_OLD_VISON 12  //ort1.12.0 old version ORT API
struct OutputSeg {
	int id;             //result id
	float confidence;   //result confidence
	cv::Rect box;       //result bounding box [w,h from point(x,y)]
	cv::Mat boxMask;       //binary box mask in bbox area
	std::vector<cv::Point> polygonPoints;     //polygon points silhouette of the mask in the whole image
};
struct MaskParams {
	int segChannels = 32;
	int segWidth = 160;
	int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.55;
	cv::Size srcImgShape;
	cv::Vec4d params;

};
bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize); 
void DrawPred(cv::Mat& img, std::vector<OutputSeg> result, std::vector<std::string> classNames, std::vector<cv::Scalar> color, const std::string& out_path);
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114));
void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<OutputSeg>& output, const MaskParams& maskParams);
void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputSeg& output, const MaskParams& maskParams);
std::vector<cv::Point> binaryMaskToPolygon(const cv::Mat& mask, const cv::Rect& box);
void check_opencv();


