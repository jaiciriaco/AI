import cv2 as cv

img1 = cv.imread("data/poster0.jpeg")
img2 = cv.imread("data/poster1.jpeg")
img3 = cv.imread("data/poster2.jpeg")

imagenes = [img1, img2, img3]

stitcher = cv.Stitcher_create(cv.Stitcher_PANORAMA)

status, pano = stitcher.stitch(imagenes)

cv.namedWindow("panorama", cv.WINDOW_NORMAL)
cv.imshow("panorama", pano)
cv.waitKey(0)
cv.destroyAllWindows()
