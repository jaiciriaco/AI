import cv2 as cv
import numpy as np

def leer_o_fallar(path):
    img = cv.imread(path)
    return img

def preprocesar(gray, modo, clahe_clip, clahe_grid, canny1, canny2):
    if modo == 0:
        return gray
    if modo == 1:
        g = max(2, int(clahe_grid))
        clahe = cv.createCLAHE(clipLimit=max(
            1.0, clahe_clip / 10.0), tileGridSize=(g, g))
        return clahe.apply(gray)
    c1 = max(0, int(canny1))
    c2 = max(c1 + 1, int(canny2))
    return cv.Canny(gray, c1, c2)

def orb_kp_desc(gray, nfeatures, fast_th):
    orb = cv.ORB_create(
        nfeatures=max(100, int(nfeatures)),
        fastThreshold=max(0, int(fast_th)),
        scoreType=cv.ORB_HARRIS_SCORE
    )
    return orb.detectAndCompute(gray, None)

def matches_ratio(des_src, des_ref, ratio):
    if des_src is None or des_ref is None:
        return []
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_src, des_ref, k=2)
    out = []
    r = max(0.10, min(0.99, float(ratio)))
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < r * n.distance:
            out.append(m)
    return out

def homografia(kps_src, kps_ref, matches, ransac_thr):
    if len(matches) < 4:
        return None, None
    pts_src = np.float32(
        [kps_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_ref = np.float32(
        [kps_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(
        pts_src, pts_ref, cv.RANSAC, max(0.5, float(ransac_thr)))
    return H, mask

def overlay(base, top, alpha):
    a = max(0.0, min(1.0, float(alpha)))
    return cv.addWeighted(base, 1.0 - a, top, a, 0)

PREPROC_MODO = 1
CLAHE_CLIP = 25
CLAHE_GRID = 8
CANNY1 = 60
CANNY2 = 180

NFEATURES = 2500
FAST_TH = 12
RATIO = 0.75
RANSAC_THR = 3.0
ALPHA = 0.35

ref = leer_o_fallar("data/formulario.png")
src = leer_o_fallar("data/formularioRelleno-original.png")

ref_g0 = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
src_g0 = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
h_ref, w_ref = ref_g0.shape

ref_g = preprocesar(ref_g0, PREPROC_MODO, CLAHE_CLIP,
                    CLAHE_GRID, CANNY1, CANNY2)
src_g = preprocesar(src_g0, PREPROC_MODO, CLAHE_CLIP,
                    CLAHE_GRID, CANNY1, CANNY2)

kps_ref, des_ref = orb_kp_desc(ref_g, NFEATURES, FAST_TH)
kps_src, des_src = orb_kp_desc(src_g, NFEATURES, FAST_TH)

m = matches_ratio(des_src, des_ref, RATIO)
H, inliers = homografia(kps_src, kps_ref, m, RANSAC_THR)


warp = cv.warpPerspective(
        src, H, (w_ref, h_ref), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))
cv.imwrite("data/formularioRelleno-alineado.png", warp)

over = overlay(ref, warp, ALPHA)
diff = cv.absdiff(ref_g0, cv.cvtColor(warp, cv.COLOR_BGR2GRAY))

cv.namedWindow("ref | warp | diff", cv.WINDOW_NORMAL)
cv.namedWindow("overlay", cv.WINDOW_NORMAL)

vista = np.hstack([ref, warp, cv.cvtColor(diff, cv.COLOR_GRAY2BGR)])
cv.imshow("ref | warp | diff", vista)
cv.imshow("overlay", over)

img_matches = cv.drawMatches(src, kps_src, ref, kps_ref,
                             m[:120], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.namedWindow("matches", cv.WINDOW_NORMAL)
cv.imshow("matches", img_matches)

cv.waitKey(0)
cv.destroyAllWindows()
