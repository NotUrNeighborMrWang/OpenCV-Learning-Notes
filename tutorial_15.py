import cv2 as cv
import numpy as np


# 图像金字塔
def pyramid_demo(image):
    level = 3
    temp =image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_demo"+str(i), dst)
        temp = dst.copy()
    return pyramid_images


# 拉普拉斯金字塔
def lapalian_demo(image):
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("lapalian_down_" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("lapalian_down_"+str(i), lpls)


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

lapalian_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
