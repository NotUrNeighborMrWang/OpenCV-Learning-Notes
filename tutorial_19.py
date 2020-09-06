"""
圆检测
"""
import cv2 as cv
import imutils
import numpy as np


def img_process(image):
    """图像处理"""
    # step 1 : 剪裁 + 灰度
    image = cv.resize(src, (620, 480))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # step 2 : 去噪
    gray = cv.bilateralFilter(gray, 13, 15, 15)
    edged = cv.Canny(gray, 30, 200)

    # step 3 : 寻找轮廓
    contours = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
    screenCnt = None
    # print("contours的数据类型为：%s " % type(contours))

    for c in contours:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv.drawContours(src, [screenCnt], -1, (0, 0, 255), 3)

    # step 4 : 遮罩
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv.bitwise_and(image, image, mask=mask)
    cv.imshow("mask_image", new_image)
    cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)

    return new_image


def detect_circles_demo(image):
    # dst = cv.pyrMeanShiftFiltering(image, 10, 100)
    cimage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv.imshow("circles", image)


print("----------Hello OpenCV----------")
src = cv.imread(r"./car_demo.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
cropped = img_process(src)
detect_circles_demo(cropped)
cv.waitKey(0)
cv.destroyAllWindows()

