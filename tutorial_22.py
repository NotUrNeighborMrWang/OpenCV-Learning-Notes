import cv2 as cv


# erode - 腐蚀
def erode_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernal = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.erode(binary, kernal)
    cv.imshow("erode_demo", dst)


# dilate - 膨胀
def dilate_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernal = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(binary, kernal)
    cv.imshow("dilate_demo", dst)


print("----------Hello OpenCV----------")
src = cv.imread("D:\\Python\\PyProjects\\OpenCV_demo\\images\\lena.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

# dilate_demo(src)

kernal = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dst = cv.erode(src, kernal)
cv.imshow("result", dst)

cv.waitKey(0)
cv.destroyAllWindows()


