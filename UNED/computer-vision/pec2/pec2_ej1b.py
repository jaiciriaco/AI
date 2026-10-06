import cv2 as cv
import numpy as np
import random

def model_from_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float(x1*y2 - x2*y1)
    n = np.sqrt(a*a + b*b) + 1e-9
    return a/n, b/n, c/n

def refit_with_inliers(inlier_xy):
    pts = np.array(inlier_xy, dtype=np.float32).reshape(-1, 1, 2)
    line = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.ravel().tolist()

    a = float(vy)
    b = float(-vx)
    c = -(a*x0 + b*y0)

    n = np.sqrt(a*a + b*b) + 1e-9
    return a/n, b/n, c/n

def draw_line(img_bgr, a, b, c, color=(0, 0, 255), thickness=2):
    h, w = img_bgr.shape[:2]
    pts = []

    if abs(b) > 1e-9:
        y0 = int(round((-c) / b))
        y1 = int(round((-c - a*(w-1)) / b))
        pts.append((0, y0))
        pts.append((w-1, y1))

    if abs(a) > 1e-9:
        x0 = int(round((-c) / a))
        x1 = int(round((-c - b*(h-1)) / a))
        pts.append((x0, 0))
        pts.append((x1, h-1))

    inside = []
    for (x, y) in pts:
        if 0 <= x < w and 0 <= y < h:
            inside.append((x, y))

    out = img_bgr.copy()
    if len(inside) >= 2:
        cv.line(out, inside[0], inside[1], color, thickness, cv.LINE_AA)
    return out

def ransac_line_vectorized(xs, ys, N, t, d):
    n = xs.shape[0]
    if n < 2:
        return None, None

    best_count = 0
    best_mask = None
    best_model = None

    for it in range(N):
        i1 = random.randrange(n)
        i2 = random.randrange(n)
        if i1 == i2:
            continue

        x1, y1 = xs[i1], ys[i1]
        x2, y2 = xs[i2], ys[i2]

        if (x1-x2)**2 + (y1-y2)**2 < 25:
            continue

        a, b, c = model_from_2_points((x1, y1), (x2, y2))

        dists = np.abs(a*xs + b*ys + c)

        mask = dists < t
        count = int(np.sum(mask))

        if count >= d and count > best_count:
            best_count = count
            best_mask = mask
            best_model = (a, b, c)

        if it % 50 == 0:
            if cv.waitKey(1) & 0xFF == 27:
                break

    if best_model is None or best_mask is None:
        return None, None

    return best_model, best_mask

img = cv.imread("data/lineaRuidosa.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.namedWindow("panel", cv.WINDOW_NORMAL)
cv.resizeWindow("panel", 420, 320)
cv.namedWindow("binaria", cv.WINDOW_NORMAL)
cv.namedWindow("vista", cv.WINDOW_NORMAL)

cv.createTrackbar("bin_th", "panel", 46, 255, lambda x: None)
cv.createTrackbar("N_iters", "panel", 1217, 5000, lambda x: None)
cv.createTrackbar("t_dist", "panel", 5, 30, lambda x: None)
cv.createTrackbar("d_inl", "panel", 532, 20000, lambda x: None)
cv.createTrackbar("max_pts", "panel", 14829, 60000, lambda x: None)

dummy = np.zeros((50, 300), dtype=np.uint8)
cv.imshow("panel", dummy)
cv.waitKey(1)

last_params = None
cached_model = None
cached_inliers = None
cached_bw = None

while True:
    if cv.getWindowProperty("panel", cv.WND_PROP_VISIBLE) < 1:
        break

    bin_th = cv.getTrackbarPos("bin_th", "panel")
    N_user = cv.getTrackbarPos("N_iters", "panel")
    t = max(1, cv.getTrackbarPos("t_dist", "panel"))
    d = max(2, cv.getTrackbarPos("d_inl", "panel"))
    max_pts = cv.getTrackbarPos("max_pts", "panel")

    params = (bin_th, N_user, t, d, max_pts)

    if params != last_params:
        last_params = params

        _, bw = cv.threshold(gray, bin_th, 255, cv.THRESH_BINARY)
        cached_bw = bw

        ys, xs = np.where(bw == 255)
        if len(xs) == 0:
            cached_model = None
            cached_inliers = None
        else:
            if len(xs) > max_pts:
                idx = np.random.choice(len(xs), size=max_pts, replace=False)
                xs_s = xs[idx].astype(np.float32)
                ys_s = ys[idx].astype(np.float32)
            else:
                xs_s = xs.astype(np.float32)
                ys_s = ys.astype(np.float32)

            N = min(N_user, 1500)

            model0, mask = ransac_line_vectorized(
                xs_s, ys_s, N=N, t=float(t), d=int(d))

            if model0 is None:
                cached_model = None
                cached_inliers = None
            else:
                inlier_xy = np.column_stack([xs_s[mask], ys_s[mask]]).tolist()
                if len(inlier_xy) >= 2:
                    a2, b2, c2 = refit_with_inliers(inlier_xy)
                    cached_model = (a2, b2, c2)
                    cached_inliers = inlier_xy
                else:
                    cached_model = None
                    cached_inliers = None

    vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    bw_show = cached_bw if cached_bw is not None else np.zeros_like(gray)

    if cached_model is not None:
        a, b, c = cached_model

        if cached_inliers is not None and len(cached_inliers) > 0:
            step = max(1, len(cached_inliers) // 6000)
            for i in range(0, len(cached_inliers), step):
                x, y = cached_inliers[i]
                x = int(x)
                y = int(y)
                if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                    vis[y, x] = (0, 255, 0)

        vis = draw_line(vis, a, b, c, color=(0, 0, 255), thickness=2)

        if abs(b) > 1e-9:
            m = -a / b
            ang = np.degrees(np.arctan(m))
        else:
            ang = 90.0

        cv.putText(vis, f"inliers={len(cached_inliers)}  ang={ang:.2f} deg",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        cv.putText(vis, "Sin modelo (baja d, sube t, baja bin_th)",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv.imshow("binaria", bw_show)
    cv.imshow("vista", vis)

    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()
