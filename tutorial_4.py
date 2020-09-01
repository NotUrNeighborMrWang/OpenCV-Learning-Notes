import cv2 as cv
import numpy as np


# 图像加法
def add_demo(m1, m2):
    # OpenCV加法是饱和运算，而Numpy加法是模运算。
    dst = cv.add(m1, m2)
    cv.imshow("add_demo", dst)


# 图像减法
def subtract_demo(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", dst)


# 图像除法
def divide_demo(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow("divide_demo", dst)


# 图像乘法
def multiply_demo(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow("multiply_demo", dst)


# 逻辑运算
def logic_demo(m1, m2):
    # dst = cv.bitwise_and(m1, m2)
    dst = cv.bitwise_or(m1, m2)
    cv.imshow("logic_demo", dst)


# 亮度和对比度
def contrast_brightness_demo(image, c, b):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow("con-bri-demo", dst)


# 均值和方差
def others(m1, m2):
    M1, dev1 = cv.meanStdDev(m1)
    M2, dev2 = cv.meanStdDev(m2)
    h, w = m1.shape[:2]

    print("-"*20)
    print("图1的均值%s：\n", M1)
    print("图2的均值%s：\n", M2)

    print("-"*20)
    print("图1的方差%s：\n", dev1)
    print("图2的方差%s：\n", dev2)

    img = np.zeros([h, w], np.uint0)
    m, dev = cv.meanStdDev(img)
    print("img平均值%s:\n", m)
    print("img方差%s:\n", dev)


print("----------Hello OpenCV----------")
# src1 = cv.imread(r"D:/python/PyProject/OpenCV_demo/images/LinuxLogo.jpg")
# src2 = cv.imread(r"D:/python/PyProject/OpenCV_demo/images//WindowsLogo.jpg")
# print(src1.shape)
# print(src2.shape)
# cv.namedWindow("image1", cv.WINDOW_AUTOSIZE)
# cv.imshow("image1", src1)       # 在窗口中显示图像，窗口自动适合图像尺寸
# cv.imshow("image2", src2)       # 在窗口中显示图像，窗口自动适合图像尺寸

# add_demo(src1, src2)
# subtract_demo(src1, src2)
# divide_demo(src1, src2)
# multiply_demo(src1, src2)
# others(src1,src2)
# logic_demo(src1,src2)
src = cv.imread("./demo2.jpg")
cv.imshow("demo2", src)
contrast_brightness_demo(src, 1.5, 10)
cv.waitKey(0)       # 键盘绑定函数，等待任何键盘事件指定的毫秒

cv.destroyAllWindows()
