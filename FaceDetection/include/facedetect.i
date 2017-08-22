%module pyfacedetect
%{
#include "face_detection.h"
#include "common.h"
%}
#define SEETA_API
#define OPENCV 2.4.3
%include "face_detection.h"
%include "common.h"