"""
仿射变换
"""
import cv2 as cv
import numpy as np


def fangshe(image):
	rows, cols, ch = image.shape
	pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
	pts2 = np.float32([[10, 100], [200, 50], [100, 200]])
	M = cv.getAffineTransform(pts1, pts2)
	dst = cv.warpAffine(image, M, (cols, rows))
	cv.imshow("Input", image)
	cv.imshow("Output", dst)


print("----------Hello OpenCV----------")

img = cv.imread("demo2.jpg")

fangshe(img)

cv.waitKey(0)

cv.destroyAllWindows()