/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Detection module, containing codes implementing the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include "face_detection.h"

#include <memory>
#include <vector>

#include "detector.h"
#include "fust.h"
#include "util/image_pyramid.h"
 
#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#endif
 
namespace seeta {

class FaceDetection::Impl {
 public:
  Impl()
      : detector_(new seeta::fd::FuStDetector()),
        slide_wnd_step_x_(4), slide_wnd_step_y_(4),
        min_face_size_(20), max_face_size_(-1),
        cls_thresh_(3.85f) {}

  ~Impl() {}

  inline bool IsLegalImage(const seeta::ImageData & image) {
    return (image.num_channels == 1 && image.width > 0 && image.height > 0 &&
      image.data != nullptr);
  }

 public:
  static const int32_t kWndSize = 40;

  int32_t min_face_size_;
  int32_t max_face_size_;
  int32_t slide_wnd_step_x_;
  int32_t slide_wnd_step_y_;
  float cls_thresh_;

  std::vector<seeta::FaceInfo> pos_wnds_;
  std::unique_ptr<seeta::fd::Detector> detector_;
  seeta::fd::ImagePyramid img_pyramid_;
};

FaceDetection::FaceDetection(const char* model_path)
    : impl_(new seeta::FaceDetection::Impl()) {
  impl_->detector_->LoadModel(model_path);
}

FaceDetection::~FaceDetection() {
  if (impl_ != nullptr)
    delete impl_;
}

std::vector<seeta::FaceInfo> FaceDetection::Detect(
    const seeta::ImageData & img) {
  if (!impl_->IsLegalImage(img))
    return std::vector<seeta::FaceInfo>();

  int32_t min_img_size = img.height <= img.width ? img.height : img.width;
  min_img_size = (impl_->max_face_size_ > 0 ?
    (min_img_size >= impl_->max_face_size_ ? impl_->max_face_size_ : min_img_size) :
    min_img_size);

  impl_->img_pyramid_.SetImage1x(img.data, img.width, img.height);
  impl_->img_pyramid_.SetMinScale(static_cast<float>(impl_->kWndSize) / min_img_size);

  impl_->detector_->SetWindowSize(impl_->kWndSize);
  impl_->detector_->SetSlideWindowStep(impl_->slide_wnd_step_x_,
    impl_->slide_wnd_step_y_);

  impl_->pos_wnds_ = impl_->detector_->Detect(&(impl_->img_pyramid_));

  for (int32_t i = 0; i < impl_->pos_wnds_.size(); i++) {
    if (impl_->pos_wnds_[i].score < impl_->cls_thresh_) {
      impl_->pos_wnds_.resize(i);
      break;
    }
  }

  return impl_->pos_wnds_;
}

void FaceDetection::SetMinFaceSize(int32_t size) {
  if (size >= 20) {
    impl_->min_face_size_ = size;
    impl_->img_pyramid_.SetMaxScale(impl_->kWndSize / static_cast<float>(size));
  }
}

void FaceDetection::SetMaxFaceSize(int32_t size) {
  if (size >= 0)
    impl_->max_face_size_ = size;
}

void FaceDetection::SetImagePyramidScaleFactor(float factor) {
  if (factor >= 0.01f && factor <= 0.99f)
    impl_->img_pyramid_.SetScaleStep(static_cast<float>(factor));
}

void FaceDetection::SetWindowStep(int32_t step_x, int32_t step_y) {
  if (step_x > 0)
    impl_->slide_wnd_step_x_ = step_x;
  if (step_y > 0)
    impl_->slide_wnd_step_y_ = step_y;
}

void FaceDetection::SetScoreThresh(float thresh) {
  if (thresh >= 0)
    impl_->cls_thresh_ = thresh;
}

#ifdef OPENCV
// using namespace cv;
// cv::Rect MergeRect(std::vector<seeta::FaceInfo>& faces)
// {

//       int32_t lx = INT_MAX, ly = INT_MAX;
//       int32_t rx = 0, ry = 0;
//       int32_t num_face = static_cast<int32_t>(faces.size());

//       for (int32_t i = 0; i < num_face; i++) {
//         lx = min(lx, faces[i].bbox.x);
//         ly = min(ly, faces[i].bbox.y);
//         rx = max(rx, faces[i].bbox.x + faces[i].bbox.width);
//         ry = max(ry, faces[i].bbox.y + faces[i].bbox.height);
//       }
      
//       cv::Rect face_rect(lx, ly, rx - lx + 1, ry - ly + 1);
//       return face_rect;
// }

template <>
std::vector<cv::Rect> detect(cv::Mat& img, std::string model_path)
{
  cv::Mat img_gray;

  if (img.channels() != 1)
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  else
    img_gray = img;

  seeta::FaceDetection detector(model_path);
  int min_size = std::max(30, std::min(img.rows, img.cols) / 12);
  detector.SetMinFaceSize(min_size);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  seeta::ImageData img_data;
  img_data.data = img_gray.data;
  img_data.width = img_gray.cols;
  img_data.height = img_gray.rows;
  img_data.num_channels = 1;

  std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
  int32_t num_face = static_cast<int32_t>(faces.size());
  std::vector<cv::Rect> cvfaces;
  for (int32_t i = 0; i < num_face; i++) {
    cv::Rect rect(faces[i].bbox.x, faces[i].bbox.y, faces[i].bbox.width, faces[i].bbox.height);
    cvfaces.push_back(rect);
  }
  return cvfaces;
}
template <>
std::vector<cv::Rect> detect(string& filename)
{
  seeta::FaceDetection detector(model_path);
  cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  return detect(img);
}
#endif
}  // namespace seeta
