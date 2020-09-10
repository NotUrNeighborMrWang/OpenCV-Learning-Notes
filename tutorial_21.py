import cv2 as cv
import numpy as np


def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    print("threshold value : %s" % ret)     # 打印阈值
    cv.imshow("binary image", binary)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)        # 得到轮廓的外接矩形
        rate = min(w, h)/max(w, h)
        print("rectangle rate : %s" % rate)
        mm = cv.moments(contour)
        print(type(mm))
        cx = mm['m10']/mm['m00']
        cy = mm['m01']/mm['m00']
        cv.circle(image, (np.int(cx), np.int(cy)), 2, (0, 0, 255), -1)
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("contour area : %s" % area)
        approxCurve = cv.approxPolyDP(contour, 4, closed=True)
        print(approxCurve.shape)
        if approxCurve.shape[0] > 10:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)
        if approxCurve.shape[0] == 4:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
        if approxCurve.shape[0] == 3:
            cv.drawContours(dst, contours, i, (255, 0, 0), 2)

    cv.imshow("measure_contours", dst)


print("----------Hello OpenCV----------")
src = cv.imread("D:\\Python\\PyProjects\\OpenCV_demo\\images\\pic1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

measure_object(src)

cv.waitKey(0)

cv.destroyAllWindows()

