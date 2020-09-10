import cv2 as cv
import numpy as np


def top_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernal = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernal)
    cimage = np.array(gray.shape, np.uint8)
    cimage = 150
    dst = cv.add(dst, cimage)
    cv.imshow("top_hat", dst)


def hat_binary_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)
    cv.imshow("gray_hat", dst)


print("----------Hello OpenCV----------")
src = cv.imread("D:\\Python\\PyProjects\\OpenCV_demo\\images\\demo2.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

hat_binary_demo(src)

cv.waitKey(0)

cv.destroyAllWindows()


