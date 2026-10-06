import cv2 as cv
import math

zigzag = cv.imread("data/zigzag.jpg", cv.IMREAD_COLOR)
brain = cv.imread("data/brainLabels.png",  cv.IMREAD_COLOR)

size = (200, 200)

zigzag_200 = cv.resize(zigzag, size, interpolation=cv.INTER_LINEAR)
brain_200 = cv.resize(brain,  size, interpolation=cv.INTER_LINEAR)


def rotate_keep_bounds(img, angle_deg):
    h, w = img.shape[:2]
    angle = math.radians(angle_deg)

    cx, cy = w / 2.0, h / 2.0

    cos_a = abs(math.cos(angle))
    sin_a = abs(math.sin(angle))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(h * cos_a + w * sin_a)

    M = cv.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    tx = (new_w / 2.0) - cx
    ty = (new_h / 2.0) - cy
    M[0, 2] += tx
    M[1, 2] += ty

    rotated = cv.warpAffine(img, M, (new_w, new_h),
                            flags=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    return rotated


zigzag_rot = rotate_keep_bounds(zigzag_200, 45)
brain_rot = rotate_keep_bounds(brain_200,  45)

cv.imshow("zigzag 200x200", zigzag_200)
cv.imshow("zigzag rotado 45", zigzag_rot)

cv.imshow("brain 200x200", brain_200)
cv.imshow("brain rotado 45", brain_rot)

print("Pulsa una tecla para salir")
cv.waitKey(0)
cv.destroyAllWindows()
