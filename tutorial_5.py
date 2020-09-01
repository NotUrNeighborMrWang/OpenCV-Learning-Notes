"""
ROI与泛洪填充（Region of Interest）
"""

import cv2 as cv
import numpy as np


def fill_color_demo(image):
    copyImg = image.copy()
    h, w = image[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    cv.floorFill


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

face = src[200:400, 200:350]
gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
src[200:400, 200:350] = backface
cv.imshow("face", src)
cv.waitKey(0)

cv.destroyAllWindows()
