import cv2 as cv
import numpy as np

img = cv.imread("data/textoMolinos.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
H, W = gray.shape

cv.namedWindow("panel", cv.WINDOW_NORMAL)
cv.resizeWindow("panel", 420, 0)

cv.namedWindow("vista", cv.WINDOW_NORMAL)

cv.createTrackbar("hough_th", "panel", 0, 400, lambda x: None)
cv.createTrackbar("angle_offset", "panel", 17, 20, lambda x: None)
cv.createTrackbar("thresh", "panel", 130, 255, lambda x: None)
cv.createTrackbar("min_area", "panel", 23, 3000, lambda x: None)
cv.createTrackbar("max_area", "panel", 152, 20000, lambda x: None)
cv.createTrackbar("norm_mode", "panel", 1, 1, lambda x: None)

dummy = np.zeros((50, 300), dtype=np.uint8)
cv.imshow("panel", dummy)
cv.waitKey(1)

while True:
    if cv.getWindowProperty("panel", cv.WND_PROP_VISIBLE) < 1:
        break

    hough_th = cv.getTrackbarPos("hough_th", "panel")
    angle_offset = cv.getTrackbarPos("angle_offset", "panel") - 10
    thresh = cv.getTrackbarPos("thresh", "panel")
    min_area = cv.getTrackbarPos("min_area", "panel")
    max_area = cv.getTrackbarPos("max_area", "panel")
    norm_mode = cv.getTrackbarPos("norm_mode", "panel")

    edges = cv.Canny(gray, 0, 0)

    lines = cv.HoughLines(edges, 1, np.pi / 180, max(1, hough_th))

    angles = []
    if lines is not None:
        for l in lines:
            rho, theta = l[0]
            angle = np.degrees(theta) - 90.0
            if abs(angle) < 45:
                angles.append(angle)

    base_angle = np.median(angles) if len(angles) > 0 else 0.0
    text_angle = base_angle + angle_offset

    M = cv.getRotationMatrix2D((W//2, H//2), -text_angle, 1.0)
    rotated = cv.warpAffine(
        img, M, (W, H), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))

    rot_gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
    if thresh == 0:
        bw = cv.threshold(rot_gray, 0, 255,
                          cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    else:
        bw = cv.threshold(rot_gray, thresh, 255, cv.THRESH_BINARY_INV)[1]

    num, labels, stats, _ = cv.connectedComponentsWithStats(bw)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if min_area < area < max_area:
            boxes.append((x, y, w, h))

    letters = []
    for (x, y, w, h) in boxes:
        letter = bw[y:y+h, x:x+w]

        if norm_mode == 0:
            norm = cv.resize(letter, (20, 20), interpolation=cv.INTER_NEAREST)
        else:
            hh, ww = letter.shape
            scale = min(20/ww, 20/hh)
            nw = max(1, int(ww * scale))
            nh = max(1, int(hh * scale))

            resized = cv.resize(
                letter, (nw, nh), interpolation=cv.INTER_NEAREST)
            norm = np.zeros((20, 20), np.uint8)
            x0 = (20 - nw)//2
            y0 = (20 - nh)//2
            norm[y0:y0+nh, x0:x0+nw] = resized

        letters.append(norm)

    if len(letters) > 0:
        cols = 20
        rows = int(np.ceil(len(letters)/cols))
        collage = np.zeros((rows*20, cols*20), np.uint8)
        for i, l in enumerate(letters):
            r = i // cols
            c = i % cols
            collage[r*20:(r+1)*20, c*20:(c+1)*20] = l
    else:
        collage = np.zeros((20, 20), np.uint8)

    vis = rotated.copy()
    for (x, y, w, h) in boxes:
        cv.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv.putText(vis, f"Hough={base_angle:.2f} | offset={angle_offset:.1f} | final={text_angle:.2f}",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    canvas = np.hstack([
        cv.cvtColor(gray, cv.COLOR_GRAY2BGR),
        cv.cvtColor(edges, cv.COLOR_GRAY2BGR),
        vis
    ])

    cv.imshow("vista", canvas)
    cv.imshow("letras 20x20", collage)

    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()
