import cv2 as cv


def contours_demo(images):
    dst = cv.GaussianBlur(images, (3, 3), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary image", binary)

    contours, heriachy = cv.findContours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contours in enumerate(contours):
        cv.drawContours(images, contours, i, (0, 0, 255), -1)
        print(i)
    cv.imshow("detect contours", images)


print("----------Hello OpenCV----------")
src = cv.imread("D:\\Python\\PyProjects\\OpenCV_demo\\images\\demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

contours_demo(src)

cv.waitKey(0)

cv.destroyAllWindows()
