import cv2 as cv
import numpy as np
import math

BLUR_K = 5

MORPH_K = 3
OPEN_IT = 2
CLOSE_IT = 0

MIN_AREA = 80
MAX_AREA = 150000

SOLIDITY_MIN = 0.96
CIRC_CIRCLE_MIN = 0.86
AR_CIRCLE_MAX = 1.25

AR_ELLIPSE_MIN = 1.35
CIRC_ELLIPSE_MIN = 0.50

EPS_FRAC_PERIM = 0.02

LAST_ROW_PERCENTILE = 80

SAVE_OVERLAY = True
OVERLAY_PATH = "data/ej3a_overlay.png"

def k_odd(v):
    return v if (v % 2 == 1) else (v + 1)

def safe_centroid(cnt):
    m = cv.moments(cnt)
    if abs(m["m00"]) < 1e-6:
        x, y, w, h = cv.boundingRect(cnt)
        return (x + w // 2, y + h // 2)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return (cx, cy)

def color_gris_por_v(v):
    if v < 50:
        return "negro"
    if v < 130:
        return "gris_oscuro"
    if v < 180:
        return "gris_medio"
    return "gris_claro"

def missing_pct_enclosing_circle(cnt):
    a = float(cv.contourArea(cnt))
    (_, _), r = cv.minEnclosingCircle(cnt)
    full = math.pi * (r * r)
    if full < 1e-6:
        return None
    pct = (1.0 - a / full) * 100.0
    return max(0.0, min(100.0, pct))

def approx_poly_sides(cnt, eps_frac):
    per = float(cv.arcLength(cnt, True))
    approx = cv.approxPolyDP(cnt, eps_frac * per, True)
    return int(len(approx)), approx

def classify_shape(cnt, cy, y_thr,
                   solidity_min, circ_circle_min, ar_circle_max,
                   ar_ellipse_min, circ_ellipse_min):
    area = float(cv.contourArea(cnt))
    perim = float(cv.arcLength(cnt, True))
    circ = (4.0 * math.pi * area) / (perim * perim + 1e-6)

    hull = cv.convexHull(cnt)
    hull_area = float(cv.contourArea(hull))
    sol = (area / hull_area) if hull_area > 1e-6 else 0.0

    axis_ratio = None
    if len(cnt) >= 5:
        e = cv.fitEllipse(cnt)
        (_, _), (MA, ma), _ = e
        axis_ratio = float(max(MA, ma) / max(1e-6, min(MA, ma)))

    ar = axis_ratio if axis_ratio is not None else 999.0
    is_last_row = (cy > y_thr)

    if is_last_row and ar <= 1.30:
        shape = "circulo"
    elif sol >= solidity_min and circ >= circ_circle_min and ar <= ar_circle_max:
        shape = "circulo"
    elif sol >= solidity_min and ar >= ar_ellipse_min and circ >= circ_ellipse_min:
        shape = "elipse"
    else:
        shape = "otro"

    return shape, circ, sol, axis_ratio, is_last_row

img = cv.imread("data/blob.jpg")

H, W = img.shape[:2]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

BLUR_K = k_odd(BLUR_K)
MORPH_K = k_odd(MORPH_K)

border = np.concatenate([
    gray[:10, :].ravel(),
    gray[-10:, :].ravel(),
    gray[:, :10].ravel(),
    gray[:, -10:].ravel()
])
bg_med = int(np.median(border))
thr = max(0, min(255, bg_med - 10))

g = cv.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
bw = cv.threshold(g, thr, 255, cv.THRESH_BINARY_INV)[1]

ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
if OPEN_IT > 0:
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, ker, iterations=OPEN_IT)
if CLOSE_IT > 0:
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, ker, iterations=CLOSE_IT)

nlab, lab, stats, _ = cv.connectedComponentsWithStats(bw, connectivity=8)

valid = []
cys = []

for lbl in range(1, nlab):
    area_cc = stats[lbl, cv.CC_STAT_AREA]
    if area_cc < MIN_AREA or area_cc > MAX_AREA:
        continue

    mask = (lab == lbl).astype(np.uint8) * 255
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        continue
    cnt = max(cnts, key=cv.contourArea)
    if cv.contourArea(cnt) < 1e-6:
        continue

    cx, cy = safe_centroid(cnt)
    valid.append((lbl, mask, cnt, cx, cy))
    cys.append(cy)


y_thr = float(np.percentile(np.array(cys, dtype=np.float32), LAST_ROW_PERCENTILE))

overlay = img.copy()
cv.line(overlay, (0, int(y_thr)), (W - 1, int(y_thr)), (0, 255, 255), 2)

resultados = []

for lbl, mask, cnt, cx, cy in valid:
    area = float(cv.contourArea(cnt))
    perim = float(cv.arcLength(cnt, True))

    shape, circularity, solidity, axis_ratio, is_last_row = classify_shape(
        cnt, cy, y_thr,
        SOLIDITY_MIN, CIRC_CIRCLE_MIN, AR_CIRCLE_MAX,
        AR_ELLIPSE_MIN, CIRC_ELLIPSE_MIN
    )

    v = cv.split(hsv)[2]
    v_mean = cv.mean(v, mask=mask)[0]
    color = color_gris_por_v(v_mean)

    missing_pct = missing_pct_enclosing_circle(cnt) if is_last_row else None

    if shape in ("circulo", "elipse") and not is_last_row:
        straight_edges = 0
        approx = None
    elif is_last_row:
        straight_edges = 2
        approx = None
    else:
        straight_edges, approx = approx_poly_sides(cnt, EPS_FRAC_PERIM)

    resultados.append({
        "id": int(lbl),
        "centro": (int(cx), int(cy)),
        "area": float(area),
        "perim": float(perim),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "axis_ratio": None if axis_ratio is None else float(axis_ratio),
        "shape": shape,
        "color": color,
        "straight_edges": int(straight_edges),
        "missing_pct": None if missing_pct is None else float(missing_pct),
        "is_last_row": bool(is_last_row),
        "cnt": cnt,
        "approx": approx
    })

areas_round = [r["area"] for r in resultados if r["shape"] in ("circulo", "elipse")]
if len(areas_round) >= 3:
    t1 = float(np.percentile(np.array(areas_round, dtype=np.float32), 33))
    t2 = float(np.percentile(np.array(areas_round, dtype=np.float32), 66))
else:
    t1, t2 = 2500.0, 9000.0

for r in resultados:
    if r["area"] < t1:
        r["size"] = "S"
    elif r["area"] < t2:
        r["size"] = "M"
    else:
        r["size"] = "L"

for r in resultados:
    cnt = r["cnt"]
    cv.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)

    if r["shape"] == "circulo":
        (ccx, ccy), rr = cv.minEnclosingCircle(cnt)
        cv.circle(overlay, (int(ccx), int(ccy)), int(rr), (0, 0, 255), 2)

    if r["shape"] == "elipse" and len(cnt) >= 5:
        e = cv.fitEllipse(cnt)
        cv.ellipse(overlay, e, (255, 0, 0), 2)

    if r["approx"] is not None and len(r["approx"]) >= 3:
        cv.polylines(overlay, [r["approx"]], True, (255, 255, 0), 2)

print("Segmentación (fondo):")
print(f"mediana borde = {bg_med}")
print(f"umbral thr    = {thr}  (objeto si gray < thr)")
print(f"MORPH: k={MORPH_K}, OPEN_IT={OPEN_IT}, CLOSE_IT={CLOSE_IT}\n")

print("Informe:")
print(f"Total blobs: {len(resultados)}")
print(f"Y umbral última fila {LAST_ROW_PERCENTILE}: {y_thr:.1f}\n")

by_shape = {}
by_color = {}
by_size = {}

for r in resultados:
    by_shape[r["shape"]] = by_shape.get(r["shape"], 0) + 1
    by_color[r["color"]] = by_color.get(r["color"], 0) + 1
    by_size[r["size"]] = by_size.get(r["size"], 0) + 1

print("Conteo por forma:")
for k in sorted(by_shape.keys()):
    print(f"  - {k}: {by_shape[k]}")

print("\nConteo por tamaño:")
for k in sorted(by_size.keys()):
    print(f"  - {k}: {by_size[k]}")

print("\nConteo por color:")
for k in sorted(by_color.keys()):
    print(f"  - {k}: {by_color[k]}")

print("\nDetalle por blob:")
print(" id | centro(x,y) | area | circ | sol | ar  | forma  | tam | color       | lados | cuña%")
print("----+-------------+------+------+-----+-----+--------+-----+-------------+-------+------")

for r in sorted(resultados, key=lambda d: d["id"]):
    ar = r["axis_ratio"]
    ar_str = " - " if ar is None else f"{ar:4.2f}"
    miss = r["missing_pct"]
    miss_str = " - " if miss is None else f"{miss:5.1f}"

    print(f"{r['id']:3d} | ({r['centro'][0]:4d},{r['centro'][1]:4d}) | "
          f"{r['area']:5.0f} | {r['circularity']:.2f} | {r['solidity']:.2f} | {ar_str:>4} | "
          f"{r['shape']:^6} | {r['size']:^3} | {r['color']:^11} | "
          f"{r['straight_edges']:5d} | {miss_str}")


bw_bgr = cv.cvtColor(bw, cv.COLOR_GRAY2BGR)
vista = np.hstack([img, bw_bgr, overlay])

cv.namedWindow("vista", cv.WINDOW_NORMAL)
cv.imshow("vista", vista)

if SAVE_OVERLAY:
    cv.imwrite(OVERLAY_PATH, overlay)
    print(f"Overlay guardado en: {OVERLAY_PATH}")

cv.waitKey(0)
cv.destroyAllWindows()
