import cv2 as cv
import numpy as np


print("----------Hello OpenCV----------")
src = cv.imread(r"./demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
cv.waitKey(0)

cv.destroyAllWindows()


"""
# 使用Matplotlib
import numpy as np
from matplotlib import pyplot as plt


src = cv.imread(r"./demo2.jpg", 0)
plt.imshow(src, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])      # 隐藏 x 轴和 y 轴上的刻度值
plt.show()
"""
