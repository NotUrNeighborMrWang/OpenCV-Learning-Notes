"""
图像二值化
"""

import cv2 as cv
import numpy as np


# 图像二值化 - 全局阈值
def threshold_demo(image):
    # 灰度处理
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 调用二值化方法：
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)     # 截断
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)    # 自己指定阈值
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

    print("threshold_value %s" % ret)
    cv.imshow("binary", binary)


def local_threshold(image):
    # 灰度处理
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 调用二值化方法：
    # dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)      # 高斯

    cv.imshow("local_threshold", dst)


def custom_threshold(image):
    # 灰度处理
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])
    mean = m.sum() / (w*h)
    print("mean", mean)

    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)

    cv.imshow("binary", binary)


# 超大图像二值化
def big_image_binary(image):
    print(image.shape)
    cw = 256
    ch = 256
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:cw+col]
            print(np.std(roi), np.mean(roi))
            dev = np.std(roi)
            # 小于15过滤
            if dev < 15:
                gray[row:row + ch, col:col + cw] = 255
            else:
                ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                gray[row:row + ch, col:cw + col] = dst
    cv.imwrite("./result_1.png", gray)


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo1.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)

big_image_binary(src)

cv.waitKey(0)
cv.destroyAllWindows()
