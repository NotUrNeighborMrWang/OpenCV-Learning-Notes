"""
tutorial_1 - 图片加载与保存
"""


import cv2 as cv
import numpy as np


# 摄像头视频读取操作
def video_demo():
    capture = cv.VideoCapture(0)
    while(True):
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)       # 视频镜像调换
        cv.imshow("video", frame)
        c = cv.waitKey(50)
        if c == 27:                     # 等待ESC退出
            break


def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)      # 数值等于shape输出相乘
    print(image.dtype)     # 打印字节位数
    pixel_data = np.array(image)
    print(pixel_data)


print("----------Hello OpenCV----------")
# src = cv.imread(r"D:/Python/PyProjects/OpenCV_demo/demo1.jpg")
src = cv.imread(r"./demo2.jpg")
# src = cv.imread(r"./demo2.jpg",0)                 ＃加载彩色灰度图像
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)   # cv.WINDOW_AUTOSIZE 自动调整窗口大小
cv.imshow("input image", src)                       # 在窗口中显示图像，窗口自动适合图像尺寸
get_image_info(src)
# gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)      # 转化为灰度图像
# cv.imwrite("./result.png", gray)                # 保存为png图片
# video_demo()
cv.waitKey(0)               # 键盘绑定函数，等待任何键盘事件指定的毫秒

cv.destroyAllWindows()


