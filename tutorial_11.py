import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# 方向映射
def back_projection_demo():
    sample = cv.imread("./result.png")
    target = cv.imread("./demo2.jpg")
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

# 2D Histogram
def hist2d_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # cv.imshow("hist2d_demo", hist)
    plt.imshow(hist, interpolation="nearest")
    plt.title("2D Histogram")
    plt.show()


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
hist2d_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()
