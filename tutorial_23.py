import cv2 as cv

def open_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernal = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernal)
    cv.imshow("open_result", binary)


print("----------Hello OpenCV----------")
src = cv.imread("D:\\Python\\PyProjects\\OpenCV_demo\\images\\demo2.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

open_demo(src)

cv.waitKey(0)

cv.destroyAllWindows()


