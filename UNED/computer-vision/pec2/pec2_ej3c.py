import cv2 as cv
import numpy as np
import time

CAMERA_INDEX = 0

WIN_STRIDE = (4, 4)
PADDING = (8, 8)
SCALE = 1.05

NMS_OVERLAP_TH = 0.65

MIN_WEIGHT = -1e9

MAX_WIDTH = 640

THICK = 2

PRINT_EVERY_SEC = 1.0

def resize_keep_aspect(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    new_w = max_w
    new_h = int(h * (new_w / float(w)))
    return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

def rect_to_xyxy(x, y, w, h):
    return (x, y, x + w, y + h)

def area_xyxy(r):
    x1, y1, x2, y2 = r
    return max(0, x2 - x1) * max(0, y2 - y1)

def overlap_ratio(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    ab = area_xyxy(b) + 1e-6
    return inter / ab

def non_max_suppression(rects_xyxy, overlap_th):

    if len(rects_xyxy) == 0:
        return []

    rects = np.array(rects_xyxy, dtype=np.float32)
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 2]
    y2 = rects[:, 3]

    idxs = np.argsort(y2)
    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(int(last))

        suppress = [len(idxs) - 1]
        for pos in range(len(idxs) - 1):
            i = idxs[pos]
            ov = overlap_ratio((x1[last], y1[last], x2[last], y2[last]),
                               (x1[i], y1[i], x2[i], y2[i]))
            if ov > overlap_th:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return [rects_xyxy[i] for i in pick]


hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cap = cv.VideoCapture(CAMERA_INDEX)

cv.namedWindow("webcam_hog_people", cv.WINDOW_NORMAL)

last_print = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_keep_aspect(frame, MAX_WIDTH)

    t0 = time.time()

    rects, weights = hog.detectMultiScale(
        frame,
        winStride=WIN_STRIDE,
        padding=PADDING,
        scale=SCALE
    )

    dt_ms = (time.time() - t0) * 1000.0

    rects_f = []
    weights_f = []
    for (r, wgt) in zip(rects, weights):
        if float(wgt) >= MIN_WEIGHT:
            x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            rects_f.append((x, y, w, h))
            weights_f.append(float(wgt))

    out = frame.copy()
    for (x, y, w, h) in rects_f:
        cv.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)

    rects_xyxy = [rect_to_xyxy(x, y, w, h) for (x, y, w, h) in rects_f]
    picks = non_max_suppression(rects_xyxy, NMS_OVERLAP_TH)

    for (x1, y1, x2, y2) in picks:
        cv.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), THICK)

    cv.imshow("webcam_hog_people", out)

    now = time.time()
    if now - last_print >= PRINT_EVERY_SEC:
        last_print = now
        if len(weights_f) > 0:
            wmin = min(weights_f)
            wmax = max(weights_f)
            print(f"Boxes: {len(rects_f)} | After NMS: {len(picks)} | dt={dt_ms:.1f} ms | w[min,max]=({wmin:.2f},{wmax:.2f})")
        else:
            print(f"Boxes: 0 | After NMS: 0 | dt={dt_ms:.1f} ms")

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
