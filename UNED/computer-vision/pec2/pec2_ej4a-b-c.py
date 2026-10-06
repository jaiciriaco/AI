import cv2 as cv
import numpy as np
import math

IMG_PATH = "data/cruces/c.png" # b y c

CLAHE_CLIP = 2.0
CLAHE_TILE = 8

FG_MIN = 0.003
FG_MAX = 0.25

DILATE_K = 3
DILATE_IT = 1

CLOSE_K = 3
CLOSE_IT = 1

OPEN_K = 3
OPEN_IT = 1

DIST_SEED_FRAC = 0.30
SEED_DILATE_K = 3
SEED_DILATE_IT = 1

BG_DILATE_K = 5
BG_DILATE_IT = 2

MIN_AREA = 400
MAX_AREA = 10_000_000

ROI_SOLID_CLOSE_K = 7
ROI_SOLID_CLOSE_IT = 2

DT_WEIGHT_POWER = 1.5

RATIO_LATINA_MIN = 1.25

THICK_BOX = 2

def k_odd(v):
    return v if (v % 2 == 1) else (v + 1)

def fg_ratio(bw):
    return float(np.count_nonzero(bw)) / float(bw.size)

def fill_holes_imfill(bw):
    h, w = bw.shape[:2]
    ff = bw.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv.floodFill(ff, mask, (0, 0), 255)
    ff_inv = cv.bitwise_not(ff)
    return cv.bitwise_or(bw, ff_inv)

def choose_otsu(gray_clahe):
    t1, bw1 = cv.threshold(gray_clahe, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    r1 = fg_ratio(bw1)

    t2, bw2 = cv.threshold(gray_clahe, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    r2 = fg_ratio(bw2)

    ok1 = (FG_MIN <= r1 <= FG_MAX)
    ok2 = (FG_MIN <= r2 <= FG_MAX)

    if ok1 and not ok2:
        return bw1, t1, "OTSU_NORMAL", r1
    if ok2 and not ok1:
        return bw2, t2, "OTSU_INV", r2
    if ok1 and ok2:
        return (bw1, t1, "OTSU_NORMAL", r1) if (r1 <= r2) else (bw2, t2, "OTSU_INV", r2)

    target = 0.5 * (FG_MIN + FG_MAX)
    return (bw1, t1, "OTSU_NORMAL(out)", r1) if (abs(r1 - target) <= abs(r2 - target)) else (bw2, t2, "OTSU_INV(out)", r2)

def preprocess(img_bgr):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    g_eq = clahe.apply(gray)

    bw0, thr, mode, ratio = choose_otsu(g_eq)

    kd = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_odd(DILATE_K), k_odd(DILATE_K)))
    kc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_odd(CLOSE_K),  k_odd(CLOSE_K)))
    ko = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_odd(OPEN_K),   k_odd(OPEN_K)))

    bw = bw0.copy()
    bw = cv.dilate(bw, kd, iterations=DILATE_IT)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kc, iterations=CLOSE_IT)
    bw = fill_holes_imfill(bw)
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, ko, iterations=OPEN_IT)

    return gray, g_eq, bw, (thr, mode, ratio)

def separate_touching_with_watershed(img_bgr, bw_final):
    dist = cv.distanceTransform(bw_final, cv.DIST_L2, 5)
    dist_max = float(dist.max()) if dist.size else 0.0

    if dist_max < 1e-6:
        h, w = bw_final.shape[:2]
        return np.zeros((h, w), dtype=np.int32), np.zeros((h, w), dtype=np.uint8)

    dist_norm = np.clip(dist / dist_max * 255.0, 0, 255).astype(np.uint8)

    _, sure_fg = cv.threshold(dist, DIST_SEED_FRAC * dist_max, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    ks = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_odd(SEED_DILATE_K), k_odd(SEED_DILATE_K)))
    sure_fg = cv.dilate(sure_fg, ks, iterations=SEED_DILATE_IT)

    kb = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_odd(BG_DILATE_K), k_odd(BG_DILATE_K)))
    sure_bg = cv.dilate(bw_final, kb, iterations=BG_DILATE_IT)

    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0

    markers_ws = markers.astype(np.int32)
    cv.watershed(img_bgr.copy(), markers_ws)

    labels = np.zeros_like(markers_ws, dtype=np.int32)
    valid = markers_ws >= 2
    labels[valid] = markers_ws[valid] - 1

    return labels, dist_norm

def solidify_roi(roi_0_255):
    ksize = k_odd(ROI_SOLID_CLOSE_K)
    pad = ksize

    roi_pad = cv.copyMakeBorder(roi_0_255, pad, pad, pad, pad,
                                borderType=cv.BORDER_CONSTANT, value=0)

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
    x = cv.morphologyEx(roi_pad, cv.MORPH_CLOSE, k, iterations=ROI_SOLID_CLOSE_IT)
    x = fill_holes_imfill(x)

    x = x[pad:pad + roi_0_255.shape[0], pad:pad + roi_0_255.shape[1]]
    return x

def weighted_principal_axes_from_dt(solid01, dist32):
    ys, xs = np.where(solid01 > 0)
    if len(xs) < 40:
        return None

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)

    d = dist32[ys, xs].astype(np.float64)
    dmax = float(d.max()) if d.size else 0.0
    if dmax < 1e-9:
        w = np.ones_like(d)
    else:
        dn = d / dmax
        w = np.power(dn, DT_WEIGHT_POWER) + 1e-9

    sw = float(np.sum(w))
    cx = float(np.sum(w * x) / sw)
    cy = float(np.sum(w * y) / sw)

    dx = x - cx
    dy = y - cy

    cxx = float(np.sum(w * dx * dx) / sw)
    cxy = float(np.sum(w * dx * dy) / sw)
    cyy = float(np.sum(w * dy * dy) / sw)

    C = np.array([[cxx, cxy],
                  [cxy, cyy]], dtype=np.float64)

    _, evecs = np.linalg.eigh(C)
    v1 = evecs[:, 1]
    v2 = evecs[:, 0]

    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)

    return (cx, cy), v1, v2

def raycast_endpoint(solid01, cx, cy, vx, vy, step=0.7, max_steps=6000):
    h, w = solid01.shape[:2]
    x = float(cx)
    y = float(cy)
    last_in = (x, y)

    for _ in range(max_steps):
        x += vx * step
        y += vy * step
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            break
        if solid01[yi, xi] == 0:
            break
        last_in = (x, y)

    return last_in

def compute_axis_length(solid01, center, v):
    cx, cy = center
    vx, vy = float(v[0]), float(v[1])

    p_plus = raycast_endpoint(solid01, cx, cy,  vx,  vy)
    p_minus = raycast_endpoint(solid01, cx, cy, -vx, -vy)

    L = math.hypot(p_plus[0] - p_minus[0], p_plus[1] - p_minus[1])
    return float(L)

def classify_by_dt_axes(roi_mask_0_255):
    solid = solidify_roi(roi_mask_0_255)
    solid01 = (solid > 0).astype(np.uint8)

    dist = cv.distanceTransform(solid01, cv.DIST_L2, 5)

    axes = weighted_principal_axes_from_dt(solid01, dist)
    if axes is None:
        return "griega"

    center, v1, v2 = axes

    L1 = compute_axis_length(solid01, center, v1)
    L2 = compute_axis_length(solid01, center, v2)

    mn = max(1e-6, min(L1, L2))
    mx = max(L1, L2)
    ratio = mx / mn

    return "latina" if ratio >= RATIO_LATINA_MIN else "griega"

img = cv.imread(IMG_PATH)

gray, g_eq, bw_final, _ = preprocess(img)
labels, dist_norm = separate_touching_with_watershed(img, bw_final)

out = img.copy()

K = int(labels.max())

for lbl in range(1, K + 1):
    mask = (labels == lbl).astype(np.uint8) * 255
    area = int(np.count_nonzero(mask))
    if area < MIN_AREA or area > MAX_AREA:
        continue

    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1

    roi = mask[y0:y0+h, x0:x0+w]
    kind = classify_by_dt_axes(roi)

    color = (0, 255, 0) if kind == "griega" else (255, 0, 0)

    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if cnts:
        cv.drawContours(out, [max(cnts, key=cv.contourArea)], -1, color, 2)
    cv.rectangle(out, (x0, y0), (x0 + w, y0 + h), color, THICK_BOX)

cv.imshow("bw_final", bw_final)
cv.imshow("dist_norm (global)", dist_norm)
cv.imshow("resultado", out)

cv.waitKey(0)
cv.destroyAllWindows()
