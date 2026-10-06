import cv2 as cv
import numpy as np

IMG_PATH = "data/cruces/d.png"

DENOISE_H = 10

ADAPTIVE_BLOCK = 31
ADAPTIVE_C = 7

THICKEN_DILATE_K = 3
THICKEN_DILATE_IT = 1

CLOSE_K = 5
CLOSE_IT = 1

OPEN_K = 3
OPEN_IT = 1

MIN_AREA = 250
MAX_AREA = 10_000_000

DT_WEIGHT_POWER = 2.0
RATIO_LATINA_MIN = 1.25

ROI_SOLID_CLOSE_K = 5
ROI_SOLID_CLOSE_IT = 1

BOX_THICK = 2

def k_odd(v: int) -> int:
    return v if (v % 2 == 1) else (v + 1)


def fill_holes_imfill(bw_0_255: np.ndarray) -> np.ndarray:
    h, w = bw_0_255.shape[:2]
    ff = bw_0_255.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv.floodFill(ff, mask, (0, 0), 255)
    ff_inv = cv.bitwise_not(ff)
    return cv.bitwise_or(bw_0_255, ff_inv)


def solidify_roi(roi_0_255: np.ndarray) -> np.ndarray:
    ksize = k_odd(ROI_SOLID_CLOSE_K)
    pad = ksize

    roi_pad = cv.copyMakeBorder(
        roi_0_255, pad, pad, pad, pad,
        borderType=cv.BORDER_CONSTANT, value=0
    )

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
    x = cv.morphologyEx(roi_pad, cv.MORPH_CLOSE, k, iterations=ROI_SOLID_CLOSE_IT)
    x = fill_holes_imfill(x)

    x = x[pad:pad + roi_0_255.shape[0], pad:pad + roi_0_255.shape[1]]
    return x


def weighted_principal_axes_from_dt(mask_0_255: np.ndarray):
    if mask_0_255 is None or mask_0_255.size == 0:
        return None, None, None, None

    bin01 = (mask_0_255 > 0).astype(np.uint8)
    if cv.countNonZero(bin01) < 50:
        return None, None, None, None

    dt = cv.distanceTransform(bin01, cv.DIST_L2, 5)
    dt_max = float(dt.max())
    if dt_max < 1e-6:
        return None, None, None, None

    dt_norm = np.clip(dt / dt_max * 255.0, 0, 255).astype(np.uint8)

    ys, xs = np.where(bin01 > 0)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

    w = (dt[ys, xs].astype(np.float64) ** DT_WEIGHT_POWER) + 1e-9
    w_sum = float(np.sum(w))
    mu = np.sum(pts * w[:, None], axis=0) / w_sum

    X = pts - mu[None, :]
    C = (X.T * w) @ X / w_sum

    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    evecs[:, 0] /= (np.linalg.norm(evecs[:, 0]) + 1e-12)
    evecs[:, 1] /= (np.linalg.norm(evecs[:, 1]) + 1e-12)

    return mu, evecs, evals, dt_norm


def raycast_to_mask(bin01: np.ndarray, p0_xy: np.ndarray, v_xy: np.ndarray, step=0.5, max_steps=6000):
    h, w = bin01.shape[:2]
    p = p0_xy.astype(np.float64).copy()

    last_inside = p.copy()
    for _ in range(max_steps):
        p = p + v_xy * step
        x, y = int(round(p[0])), int(round(p[1]))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        if bin01[y, x] == 0:
            break
        last_inside = p.copy()
    return last_inside


def axis_endpoints(mask_0_255: np.ndarray, mu_xy: np.ndarray, v_xy: np.ndarray):
    bin01 = (mask_0_255 > 0).astype(np.uint8)
    p1 = raycast_to_mask(bin01, mu_xy,  v_xy)
    p2 = raycast_to_mask(bin01, mu_xy, -v_xy)
    return p1, p2


def classify_by_axis_ratio(mask_roi_0_255: np.ndarray):
    solid = solidify_roi(mask_roi_0_255)

    mu, evecs, evals, _ = weighted_principal_axes_from_dt(solid)
    if mu is None:
        return "griega", {"L1": 1.0, "L2": 1.0, "ratio": 1.0, "ok": False}

    v1 = evecs[:, 0]
    v2 = evecs[:, 1]

    p1a, p1b = axis_endpoints(solid, mu, v1)
    p2a, p2b = axis_endpoints(solid, mu, v2)

    L1 = float(np.linalg.norm(p1a - p1b))
    L2 = float(np.linalg.norm(p2a - p2b))
    if L1 < 1e-6 or L2 < 1e-6:
        return "griega", {"L1": L1, "L2": L2, "ratio": 1.0, "ok": False}

    ratio = max(L1, L2) / max(1e-6, min(L1, L2))
    kind = "latina" if ratio >= RATIO_LATINA_MIN else "griega"

    return kind, {"L1": L1, "L2": L2, "ratio": ratio, "ok": True}

def preprocess_d(img_bgr: np.ndarray):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    gray_dn = cv.fastNlMeansDenoising(gray, None, h=DENOISE_H, templateWindowSize=7, searchWindowSize=21)

    bw = cv.adaptiveThreshold(
        gray_dn, 255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY_INV,
        blockSize=k_odd(ADAPTIVE_BLOCK),
        C=ADAPTIVE_C
    )

    kd = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_odd(THICKEN_DILATE_K), k_odd(THICKEN_DILATE_K)))
    bw = cv.dilate(bw, kd, iterations=THICKEN_DILATE_IT)

    kc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_odd(CLOSE_K), k_odd(CLOSE_K)))
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kc, iterations=CLOSE_IT)

    bw = fill_holes_imfill(bw)

    ko = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_odd(OPEN_K), k_odd(OPEN_K)))
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, ko, iterations=OPEN_IT)

    return gray_dn, bw

img = cv.imread(IMG_PATH)

gray_dn, bw_final = preprocess_d(img)

bin01 = (bw_final > 0).astype(np.uint8)
dist = cv.distanceTransform(bin01, cv.DIST_L2, 5)
dmax = float(dist.max()) if dist.size else 0.0
dist_norm = np.zeros_like(bw_final)
if dmax > 1e-6:
    dist_norm = np.clip(dist / dmax * 255.0, 0, 255).astype(np.uint8)

n, labels, stats, cents = cv.connectedComponentsWithStats(bw_final, connectivity=8)

out = img.copy()

for i in range(1, n):
    area = int(stats[i, cv.CC_STAT_AREA])
    if area < MIN_AREA or area > MAX_AREA:
        continue

    x = int(stats[i, cv.CC_STAT_LEFT])
    y = int(stats[i, cv.CC_STAT_TOP])
    w = int(stats[i, cv.CC_STAT_WIDTH])
    h = int(stats[i, cv.CC_STAT_HEIGHT])

    roi = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255

    kind, _info = classify_by_axis_ratio(roi)

    if kind == "griega":
        col = (0, 255, 0)
    else:
        col = (255, 0, 0)

    cv.rectangle(out, (x, y), (x + w, y + h), col, BOX_THICK)

cv.imshow("bw_final (d)", bw_final)
cv.imshow("dist_norm (d)", dist_norm)
cv.imshow("resultado (d): griega=verde | latina=azul", out)

cv.waitKey(0)
cv.destroyAllWindows()
