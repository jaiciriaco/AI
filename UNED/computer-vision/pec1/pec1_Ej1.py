import cv2 as cv
import numpy as np

INIT = dict(
    use_bg_sub=0,
    bg_ks=31,
    median_k=3,
    use_bilateral=1,
    bilat_d=9,
    bilat_sc=60,
    bilat_ss=60,
    th_mode=3,
    invert=0,
    th_val=128,
    block=31,
    C=5,
    do_open=0,
    do_close=1,
    morph_k=3,
    morph_shape=0,
    roi_on=0,
    roi_x=0,
    roi_y=0,
    roi_w=200,
    roi_h=120,
)


def odd_at_least(v, lo):
    v = int(v)
    if v < lo:
        v = lo
    if v % 2 == 0:
        v += 1
    return v


def bg_subtract(gray, ksize):
    k = odd_at_least(ksize, 3)
    bg = cv.GaussianBlur(gray, (k, k), 0)
    out = cv.subtract(gray, bg)
    return cv.normalize(out, None, 0, 255, cv.NORM_MINMAX)


def denoise(gray, median_k, use_bilateral, d, sc, ss):
    out = gray
    if median_k >= 3:
        out = cv.medianBlur(out, odd_at_least(median_k, 3))
    if use_bilateral:
        out = cv.bilateralFilter(out, int(max(1, d)), float(sc), float(ss))
    return out


def do_threshold(gray, mode, th, block, C, invert):
    if mode == 0:
        return gray
    typ = cv.THRESH_BINARY_INV if invert else cv.THRESH_BINARY
    if mode == 1:
        return cv.threshold(gray, int(th), 255, typ)[1]
    if mode == 2:
        return cv.threshold(gray, 0, 255, typ + cv.THRESH_OTSU)[1]
    if mode == 3:
        b = odd_at_least(block, 3)
        return cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, typ, b, int(C))
    return gray


def morph(img, do_open, do_close, k, shape):
    if not (do_open or do_close):
        return img
    shape_map = {0: cv.MORPH_ELLIPSE, 1: cv.MORPH_RECT, 2: cv.MORPH_CROSS}
    kernel = cv.getStructuringElement(shape_map.get(shape, 0),
                                      (odd_at_least(k, 1), odd_at_least(k, 1)))
    out = img
    if do_open:
        out = cv.morphologyEx(out, cv.MORPH_OPEN, kernel)
    if do_close:
        out = cv.morphologyEx(out, cv.MORPH_CLOSE, kernel)
    return out


def pipeline(gray, p):
    work = gray
    if p["use_bg_sub"]:
        work = bg_subtract(work, p["bg_ks"])
    work = denoise(work, p["median_k"], p["use_bilateral"],
                   p["bilat_d"], p["bilat_sc"], p["bilat_ss"])
    work = do_threshold(work, p["th_mode"], p["th_val"],
                        p["block"], p["C"], p["invert"])
    work = morph(work, p["do_open"], p["do_close"],
                 p["morph_k"], p["morph_shape"])
    return work


def draw_button(canvas, text, x, y, w, h):
    cv.rectangle(canvas, (x, y), (x+w, y+h), (255, 255, 255), -1)
    cv.rectangle(canvas, (x, y), (x+w, y+h), (0, 0, 0), 1)
    (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv.putText(canvas, text, (x + (w - tw)//2, y + (h + th)//2 - 2),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)


img = cv.imread("data/DibujosNPT/N_328_THS_TOTAL-ev1-h.png",
                cv.IMREAD_GRAYSCALE)

H, W = img.shape

cv.namedWindow("panel", cv.WINDOW_NORMAL)
cv.resizeWindow("panel", 420, 980)
cv.moveWindow("panel", 50, 50)

cv.createTrackbar("bg_sub",   "panel",
                  INIT["use_bg_sub"],    1, lambda v: None)
cv.createTrackbar("bg_ks",    "panel",
                  INIT["bg_ks"],       101, lambda v: None)
cv.createTrackbar("median_k", "panel",
                  INIT["median_k"],      15, lambda v: None)
cv.createTrackbar("bilateral", "panel",
                  INIT["use_bilateral"],  1, lambda v: None)
cv.createTrackbar("bilat_d",  "panel",
                  INIT["bilat_d"],       21, lambda v: None)
cv.createTrackbar("bilat_sc", "panel",
                  INIT["bilat_sc"],     200, lambda v: None)
cv.createTrackbar("bilat_ss", "panel",
                  INIT["bilat_ss"],     200, lambda v: None)

cv.createTrackbar("th_mode",  "panel",
                  INIT["th_mode"],        3, lambda v: None)
cv.createTrackbar("invert",   "panel",
                  INIT["invert"],         1, lambda v: None)
cv.createTrackbar("th_val",   "panel",
                  INIT["th_val"],       255, lambda v: None)
cv.createTrackbar("block",    "panel",
                  INIT["block"],         99, lambda v: None)
cv.createTrackbar("C(+20)",   "panel",
                  INIT["C"]+20,          40, lambda v: None)  # -20..+20

cv.createTrackbar("open",     "panel",
                  INIT["do_open"],        1, lambda v: None)
cv.createTrackbar("close",    "panel",
                  INIT["do_close"],       1, lambda v: None)
cv.createTrackbar("morph_k",  "panel",
                  INIT["morph_k"],       15, lambda v: None)
cv.createTrackbar("shape",    "panel",
                  INIT["morph_shape"],     2, lambda v: None)

cv.createTrackbar("roi_on",   "panel",
                  INIT["roi_on"],          1, lambda v: None)
cv.createTrackbar("roi_x",    "panel",
                  INIT["roi_x"],        W-1, lambda v: None)
cv.createTrackbar("roi_y",    "panel",
                  INIT["roi_y"],        H-1, lambda v: None)
cv.createTrackbar("roi_w",    "panel", min(
    INIT["roi_w"], W),   W, lambda v: None)
cv.createTrackbar("roi_h",    "panel", min(
    INIT["roi_h"], H),   H, lambda v: None)

cv.namedWindow("vista", cv.WINDOW_AUTOSIZE)
SAVE_BTN = {"x": 20, "y": 20, "w": 120, "h": 32}
last_result = None


def on_mouse(event, x, y, flags, param):
    global last_result
    bx, by, bw, bh = SAVE_BTN["x"], SAVE_BTN["y"], SAVE_BTN["w"], SAVE_BTN["h"]
    if event == cv.EVENT_LBUTTONDOWN and last_result is not None:
        if bx <= x <= bx+bw and by <= y <= by+bh:
            cv.imwrite("resultado.png", last_result)
            print("✔ Guardado: resultado.png")


cv.setMouseCallback("vista", on_mouse)

while True:
    P = dict(
        use_bg_sub=cv.getTrackbarPos("bg_sub",   "panel"),
        bg_ks=cv.getTrackbarPos("bg_ks",    "panel"),
        median_k=cv.getTrackbarPos("median_k", "panel"),
        use_bilateral=cv.getTrackbarPos("bilateral", "panel"),
        bilat_d=cv.getTrackbarPos("bilat_d",  "panel"),
        bilat_sc=cv.getTrackbarPos("bilat_sc", "panel"),
        bilat_ss=cv.getTrackbarPos("bilat_ss", "panel"),
        th_mode=cv.getTrackbarPos("th_mode",  "panel"),
        invert=cv.getTrackbarPos("invert",   "panel"),
        th_val=cv.getTrackbarPos("th_val",   "panel"),
        block=cv.getTrackbarPos("block",    "panel"),
        C=cv.getTrackbarPos("C(+20)",   "panel") - 20,
        do_open=cv.getTrackbarPos("open",     "panel"),
        do_close=cv.getTrackbarPos("close",    "panel"),
        morph_k=cv.getTrackbarPos("morph_k",  "panel"),
        morph_shape=cv.getTrackbarPos("shape",    "panel"),
    )

    P["bg_ks"] = odd_at_least(P["bg_ks"],   3)
    if P["median_k"] >= 1:
        P["median_k"] = odd_at_least(P["median_k"], 3)
    P["block"] = odd_at_least(P["block"],   3)
    P["morph_k"] = odd_at_least(P["morph_k"], 1)

    result = pipeline(img, P)

    roi_on = cv.getTrackbarPos("roi_on", "panel")
    rx = cv.getTrackbarPos("roi_x",  "panel")
    ry = cv.getTrackbarPos("roi_y",  "panel")
    rw = cv.getTrackbarPos("roi_w",  "panel")
    rh = cv.getTrackbarPos("roi_h",  "panel")
    rx = max(0, min(rx, W-1))
    ry = max(0, min(ry, H-1))
    rw = max(1, min(rw, W - rx))
    rh = max(1, min(rh, H - ry))
    if roi_on:
        result[ry:ry+rh, rx:rx+rw] = 255

    last_result = result

    orig_bgr = cv.cvtColor(img,    cv.COLOR_GRAY2BGR)
    res_bgr = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
    canvas = np.hstack([orig_bgr, res_bgr])

    cv.putText(canvas, "Original",  (10, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    cv.putText(canvas, "Resultado", (W+10, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    if roi_on:
        cv.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 1)
        cv.rectangle(canvas, (W+rx, ry), (W+rx+rw, ry+rh), (0, 0, 255), 1)
        cv.putText(canvas, "ROI", (W+rx+5, ry-5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    draw_button(canvas, "Guardar (click)",
                SAVE_BTN["x"], SAVE_BTN["y"], SAVE_BTN["w"], SAVE_BTN["h"])

    cv.imshow("vista", canvas)

    key = cv.waitKey(10) & 0xFF
    if key in (27, ord('q')):
        break
    if key == ord('s') and last_result is not None:
        cv.imwrite("resultado.png", last_result)
        print("✔ Guardado: resultado.png")

cv.destroyAllWindows()
