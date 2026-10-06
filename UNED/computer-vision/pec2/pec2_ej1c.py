import cv2 as cv
import numpy as np

BIN_TH = 22
HOUGH_TH_BIN = 203

CANNY_LOW = 50
CANNY_HIGH = 150
BLUR_K = 1
HOUGH_TH_CANNY = 80

MAX_LINES_DRAW = 1

def estimate_angle_from_hough(lines):
    if lines is None or len(lines) == 0:
        return None

    angles = []
    for l in lines:
        rho, theta = l[0]
        ang = np.degrees(theta) - 90.0
        if abs(ang) < 80:
            angles.append(ang)

    if len(angles) == 0:
        return None

    return float(np.median(angles))

def draw_hough_lines(img_bgr, lines, max_lines=1):
    out = img_bgr.copy()
    if lines is None:
        return out

    n = min(len(lines), max_lines)
    for i in range(n):
        rho, theta = lines[i][0]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))

        cv.line(out, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.LINE_AA)

    return out

img = cv.imread("data/lineaRuidosa.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.namedWindow("panel", cv.WINDOW_NORMAL)
cv.resizeWindow("panel", 360, 120)
cv.namedWindow("mask", cv.WINDOW_NORMAL)
cv.namedWindow("vista", cv.WINDOW_NORMAL)

cv.createTrackbar("use_canny", "panel", 0, 1, lambda x: None)

dummy = np.zeros((30, 200), dtype=np.uint8)
cv.imshow("panel", dummy)
cv.waitKey(1)

while True:
    if cv.getWindowProperty("panel", cv.WND_PROP_VISIBLE) < 1:
        break

    use_canny = cv.getTrackbarPos("use_canny", "panel")

    if use_canny == 1:
        k = 2 * BLUR_K + 1
        g = cv.GaussianBlur(gray, (k, k), 0)

        mask = cv.Canny(g, CANNY_LOW, CANNY_HIGH)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        mask = cv.dilate(mask, kernel, iterations=1)

        hough_th = HOUGH_TH_CANNY
        mode_txt = "Con Canny"
    else:
        _, mask = cv.threshold(gray, BIN_TH, 255, cv.THRESH_BINARY)
        hough_th = HOUGH_TH_BIN
        mode_txt = "Sin Canny"

    lines = cv.HoughLines(mask, 1, np.pi / 180.0, max(1, hough_th))
    angle = estimate_angle_from_hough(lines)

    vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    if MAX_LINES_DRAW > 0:
        vis = draw_hough_lines(vis, lines, max_lines=MAX_LINES_DRAW)

    txt = f"ang={angle:.2f} deg | modo={mode_txt}"

    cv.putText(vis, txt, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv.imshow("mask", mask)
    cv.imshow("vista", vis)

    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()
