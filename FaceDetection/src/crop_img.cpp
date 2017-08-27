#include "face_detection.h"
#include "crop_img.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <exception>
void* detectface(cv::Mat& img, const char* model_path)
{
  if(img.data == NULL)
  {
    printf("Image read failed\n");
    throw std::exception();
  }
  cv::Mat img_gray;

  if (img.channels() != 1)
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  else
    img_gray = img;

  seeta::FaceDetection detector(model_path);
  int min_size = std::max(30, std::min(img.rows, img.cols) / 12);
  //int min_size = 30;
  detector.SetMinFaceSize(min_size);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);
  printf("set detector success\n");
  seeta::ImageData img_data;
  img_data.data = img_gray.data;
  img_data.width = img_gray.cols;
  img_data.height = img_gray.rows;
  img_data.num_channels = 1;

  std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
  printf("Before return\n");
  Rects* rect = new Rects(faces);
  return (void*)rect;
  // printf("detect success\n");
  // int num_face = static_cast<int>(faces.size());
  // std::vector<cv::Rect> cvfaces;
  // for (int i = 0; i < num_face; i++) {
  //   cv::Rect rect(faces[i].bbox.x, faces[i].bbox.y, faces[i].bbox.width, faces[i].bbox.height);
  //   cvfaces.push_back(rect);
  // }
  // return cvfaces;
}
void* detectface(const char* filename, const char* model_path)
{
  cv::Mat img;
  try{
    img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    return detectface(img, model_path);
    
  }catch(std::exception& e)
  {
    printf("-- Image Not found --\n");
    throw std::exception();
  }
}
