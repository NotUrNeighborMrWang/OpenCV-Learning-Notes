"""
tutorial_3 - 图像的色彩空间
"""


import cv2 as cv
import numpy as np


def extract_object_demo():
    capture = cv.VideoCapture("./video.mp4")
    while True:
        # 逐帧捕获
        ret, frame = capture.read()
        # 如果正确读取帧，ret为True
        if not ret:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([37, 43, 46])
        upper_hsv = np.array([77, 255, 255])
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        dst = cv.bitwise_and(frame, frame, mask=mask)
        # 显示结果帧
        cv.imshow("video", frame)
        cv.imshow("mask", dst)
        c = cv.waitKey(40)
        if c == 27:
            break


# 色彩空间操作
def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)
    Ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow("Ycrcb", Ycrcb)


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo2.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)       # 自动调整窗口大小
cv.imshow("input image", src)       # 在窗口中显示图像，窗口自动适合图像尺寸

# color_space_demo(src)
extract_object_demo()
# b, g, r = cv.split(src)     # 三通道
# cv.imshow("blue", b)
# cv.imshow("green", g)
# cv.imshow("red", r)

# src[:, :, 0] = 0        # 更改通道
# src = cv.merge([b, g, r])       # cv.merge输入为一个列表
# cv.imshow("changed image", src)

cv.waitKey(0)       # 键盘绑定函数，等待任何键盘事件指定的毫秒

cv.destroyAllWindows()
