import cv2 as cv
import numpy as np


# 直方图均衡化
def equalHist_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow("equalHist_demo", dst)


# 局部自适应
def clahe_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv.imshow("clahe_demo", dst)


def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)
    bsize = 256/16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
    return rgbHist


def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离：%s, \n相关性：%s, \n卡方：%s" % (match1, match2, match3))


print("----------Hello OpenCV----------")
# src = cv.imread(r"D:/python/PyProject/OpenCV_demo/images/pic6.png")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)       # cv.WINDOW_AUTOSIZE 自动调整窗口大小
# cv.imshow("input image", src)       # 在窗口中显示图像，窗口自动适合图像尺寸

image1 = cv.imread("./result.png")
image2 = cv.imread("./demo2.jpg")
cv.imshow("image1", image1)
cv.imshow("image2", image2)
hist_compare(image1, image2)

cv.waitKey(0)       # 键盘绑定函数，等待任何键盘事件指定的毫秒

cv.destroyAllWindows()