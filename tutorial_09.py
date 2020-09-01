"""
图像直方图：histogram
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


# 三通道的值
def image_hist(image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo2.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
image_hist(src)
cv.waitKey(0)

cv.destroyAllWindows()


