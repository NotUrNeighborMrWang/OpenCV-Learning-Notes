"""

LicensePlateRecognition using OpenCV python

局限：
    1. 字符识别不准确；
    2. 一张图中有多个车牌时只识别单一车牌（中间位置的车牌)；
    3. 含中文车牌识别为乱码。

"""

import cv2 as cv
import numpy as np
import imutils
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\python\Tesseract-OCR\tesseract.exe'


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
    # cv.imshow("mask_image", new_image)

    # step 5 : 字符分割
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx+1, topy:bottomy+1]
    # cropped = cv.equalizeHist(cropped)
    cv.imshow("cropped_image", cropped)

    return cropped


def img2text(image):
    """字符识别"""
    text = pytesseract.image_to_string(image, config='--psm 11')
    print("Detected license plate Number is:", text)


if __name__ == '__main__':

    # 显示原始图片
    print("\n----------OpenCV-Python - CarDemo----------\n")
    src = cv.imread(r"./cars1.jpg")
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("original car_demo image", src)

    img = img_process(src)
    img2text(img)

    cv.waitKey(0)
    cv.destroyAllWindows()
