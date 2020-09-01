"""
边缘保留滤波
"""

import cv2 as cv


# 高斯模糊 + 边缘保留
def bi_demo(image):
    dst = cv.bilateralFilter(image, 0, 100, 15)
    cv.imshow("bi_demo", dst)


# 均值迁移
def shift_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow("shift_demo", dst)


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo2.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
bi_demo(src)        # 高斯模糊 + 边缘保留
shift_demo(src)     # 均值迁移
cv.waitKey(0)

cv.destroyAllWindows()
