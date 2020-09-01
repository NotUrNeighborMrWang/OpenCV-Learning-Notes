"""
numpy数组操作
"""

import cv2 as cv
import numpy as np


# 遍历数组中的每个像素点 + 像素取反
def access_pixels(image):
    print(image.shape)
    print(image.size)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width : %s , height : %s , channels : %s" % (width, height, channels))  # 编程技巧
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("pixels_demo", image)


# 创建新的图像
def create_image():
    """
    # 多通道图像
    img = np.zeros([400, 400, 3], np.uint8)
    img[: , : , 0] = np.ones([400, 400])*255
    cv.imshow("new image", img)

    # 单通道图像
    img = np.zeros([400, 400, 1], np.uint8)
    # img[: , : , 0] = np.ones([400, 400]) * 127
    img = img * 0
    cv.imshow("new image", img)
    cv.imwrite("./myImage.png", img)
    """

    m1 = np.ones([3, 3], np.uint8)
    m1.fill(12222.388)  # 将一个区间的元素都赋予val值
    print(m1)

    m2 = m1.reshape([1, 9])
    print(m2)

    """
    # 自定义的时候使用此方法
    m3 = np.array([[2,3,4],[4,5,6],[7,8,9]],np.int32)
    m3.fill(9)
    print(m3)
    """


# 像素取反
def inverse(image):
    dst = cv.bitwise_not(image)  # 像素取反API
    cv.imshow("inverse_demo", dst)


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo2.jpg")  # blue, green, red
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)       # cv.WINDOW_AUTOSIZE 自动调整窗口大小
# cv.imshow("input image", src)       # 在窗口中显示图像，窗口自动适合图像尺寸
t1 = cv.getTickCount()
# access_pixels(src)
create_image()      # 创建新的图像
# inverse(src)
t2 = cv.getTickCount()
time = (t2 - t1) / cv.getTickFrequency()
print("time : %s ms" % (time * 1000))
cv.waitKey(0)  # 键盘绑定函数，等待任何键盘事件指定的毫秒

cv.destroyAllWindows()
