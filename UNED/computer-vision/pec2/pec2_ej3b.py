import cv2 as cv
import time

CAMERA_INDEX = 0

FRONTAL_XML = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
PROFILE_XML = cv.data.haarcascades + "haarcascade_profileface.xml"

FRONTAL_SCALE_FACTOR = 1.1
FRONTAL_MIN_NEIGHBORS = 5
FRONTAL_MIN_SIZE = (30, 30)

PROFILE_SCALE_FACTOR = 1.1
PROFILE_MIN_NEIGHBORS = 4
PROFILE_MIN_SIZE = (30, 30)

IOU_SUPPRESS_TH = 0.35

DRAW_THICKNESS = 2

def load_cascade(xml_path):
    c = cv.CascadeClassifier(xml_path)
    return c

def preprocess(gray):
    return cv.equalizeHist(gray)

def detect_faces(gray_eq, cascade, scaleFactor, minNeighbors, minSize):
    rects = cascade.detectMultiScale(
        gray_eq,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )
    out = []
    for (x, y, w, h) in rects:
        out.append((int(x), int(y), int(w), int(h)))
    return out

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter + 1e-6
    return inter / union

def nms_rects(rects, iou_th):
    if len(rects) == 0:
        return []
    rects_sorted = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    keep = []
    for r in rects_sorted:
        ok = True
        for k in keep:
            if iou(r, k) > iou_th:
                ok = False
                break
        if ok:
            keep.append(r)
    return keep

face_frontal = load_cascade(FRONTAL_XML)
face_profile = load_cascade(PROFILE_XML)

cap = cv.VideoCapture(CAMERA_INDEX)

cv.namedWindow("webcam_haar", cv.WINDOW_NORMAL)

last_print = 0.0
PRINT_EVERY_SEC = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_eq = preprocess(gray)

    t0 = time.time()

    rects_f = detect_faces(
        gray_eq, face_frontal,
        FRONTAL_SCALE_FACTOR, FRONTAL_MIN_NEIGHBORS, FRONTAL_MIN_SIZE
    )

    rects_p = detect_faces(
        gray_eq, face_profile,
        PROFILE_SCALE_FACTOR, PROFILE_MIN_NEIGHBORS, PROFILE_MIN_SIZE
    )

    rects_final = nms_rects(rects_f + rects_p, IOU_SUPPRESS_TH)

    dt_ms = (time.time() - t0) * 1000.0

    out = frame.copy()
    for (x, y, w, h) in rects_final:
        cv.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), DRAW_THICKNESS)

    cv.imshow("webcam_haar", out)

    now = time.time()
    if now - last_print >= PRINT_EVERY_SEC:
        last_print = now
        print(f"Caras: {len(rects_final)} | tiempo detect: {dt_ms:.1f} ms")

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
