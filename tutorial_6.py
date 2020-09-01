"""
模糊操作
"""


import cv2 as cv
import numpy as np


# 均值模糊
def blur_demo(image):
	dst = cv.blur(image, (5, 5))		# 垂直方向和水平方向
	cv.imshow("blur_demo", dst)


# 中值模糊
def median_blur_demo(image):
	dst = cv.medianBlur(image, 5)		# 椒盐去噪
	cv.imshow("median_blur_demo", dst)


# 自定义模糊
def custom_blur_demo(image):
	# 椒盐去噪
	# kernel = np.ones([5, 5], np.float32) / 25

	# 轻微模糊
	# kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32) / 9

	# 锐化
	kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)

	dst = cv.filter2D(image, -1, kernel=kernel)
	cv.imshow("custom_blur_demo", dst)


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)      
cv.imshow("input image", src)   
custom_blur_demo(src)
cv.waitKey(0)      

cv.destroyAllWindows()
