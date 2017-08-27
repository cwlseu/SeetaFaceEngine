#ifndef TENCENT_CROP_IMG_H_
#define TENCENT_CROP_IMG_H_
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "common.h"
#include <exception>

typedef struct Rects{
	int num;
	cv::Rect* data;
	Rects(std::vector<seeta::FaceInfo>& faces)
	{
		num = static_cast<int>(faces.size());
		data = (cv::Rect*) malloc(sizeof(cv::Rect)*num);
		for(int i = 0; i < num; ++i)
		{
			cv::Rect rect(faces[i].bbox.x, faces[i].bbox.y, faces[i].bbox.width, faces[i].bbox.height);
			*(data+i) = rect;
		}
	}
	Rects(void* _rects)
	{
		Rects* rects = (Rects*)_rects;
		num = rects->num;
		data = (cv::Rect*) malloc(sizeof(cv::Rect)*num);
		memcpy((char*)data, (char*)rects->data, sizeof(cv::Rect)*num);
	}
	Rects(std::vector<cv::Rect>& faces)
	{
		num = static_cast<int>(faces.size());
		data = (cv::Rect*) malloc(sizeof(cv::Rect)*num);
		for(int i = 0; i < num; ++i)
		{
			*(data+i) = faces[i];
		}
	}
	cv::Rect getRect(int idx)
	{
		assert(idx < num);
		return *(data+idx);
	}
	~Rects()
	{
		if(data) 
		{
			free(data);
			data = NULL;
		}
		num = 0;
	}
}Rects;


void* detectface(cv::Mat& img, const char* modelpath);
void* detectface(const char* imgfile, const char* modelpath);
#endif