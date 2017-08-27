from pyfacedetect import detectface, Rects, Rect
import cv2
from ctypes import *

imgfile = "../../img/4.jpg"
img = cv2.imread(imgfile)

result = detectface(imgfile, "../../model/seeta_fd_frontal_v1.0.bin")
rects = Rects(result)
print rects.num
print rects.data[0]