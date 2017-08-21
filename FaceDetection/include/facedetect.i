%module pyfacedetect
%{
#include "face_detection.h"
#include "common.h"
%}
#define SEETA_API
%include "face_detection.h"
%include "common.h"