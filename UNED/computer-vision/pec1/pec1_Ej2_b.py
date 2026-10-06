# b)
import cv2 as cv
import numpy as np
import math

img = cv.imread("data/zigzag.jpg", cv.IMREAD_GRAYSCALE)

H, W = img.shape


def mat_scale(sx, sy):
    return np.float32([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ])


def mat_shear_x(angle_deg):
    t = math.tan(math.radians(angle_deg))
    return np.float32([
        [1, t, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])


def mat_rotate(angle_deg):
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    return np.float32([
        [c,  s, 0],
        [-s,  c, 0],
        [0,  0, 1]
    ])


def mat_translate(tx, ty):
    return np.float32([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])


def apply_affine(img, M, out_size):
    return cv.warpAffine(
        img.astype(np.float32),
        M[:2, :],
        out_size,
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0
    ).astype(np.uint8)


def transformed_bounds(width, height, M):
    corners = np.array([
        [0,      0,       1],
        [width,  0,       1],
        [0,      height,  1],
        [width,  height,  1]
    ], dtype=np.float32).T

    trans = M @ corners
    xs = trans[0, :]
    ys = trans[1, :]

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    return min_x, max_x, min_y, max_y


# b1)
sx = 0.25
sy = 0.25

M_T1 = mat_scale(sx, sy)

W_base, H_base = img.shape[1], img.shape[0]
W_T1 = int(W_base * sx)
H_T1 = int(H_base * sy)

img_T1 = apply_affine(img, M_T1, (W_T1, H_T1))

cv.imshow("T1 - Escala 1/4", img_T1)
cv.waitKey(0)
cv.destroyAllWindows()

angle_shear = -30
M_shear = mat_shear_x(angle_shear)

h1, w1 = img_T1.shape[:2]

min_x, max_x, min_y, max_y = transformed_bounds(w1, h1, M_shear)

tx = -min_x if min_x < 0 else 0.0
ty = -min_y if min_y < 0 else 0.0

M_T2 = mat_translate(tx, ty) @ M_shear

out_w2 = int(np.ceil(max_x - min_x))
out_h2 = int(np.ceil(max_y - min_y))

img_T2 = apply_affine(img_T1, M_T2, (out_w2, out_h2))

cv.imshow("T2 - Shear -30º", img_T2)
cv.waitKey(0)
cv.destroyAllWindows()

angle_rot = 90

h2, w2 = img_T2.shape[:2]
cx, cy = w2 / 2.0, h2 / 2.0

M_center_to_origin = mat_translate(-cx, -cy)

M_rot = mat_rotate(angle_rot)

M_origin_to_center = mat_translate(cx, cy)

M_T3_center = M_origin_to_center @ M_rot @ M_center_to_origin

min_x3, max_x3, min_y3, max_y3 = transformed_bounds(w2, h2, M_T3_center)

tx3 = -min_x3 if min_x3 < 0 else 0.0
ty3 = -min_y3 if min_y3 < 0 else 0.0

M_T3 = mat_translate(tx3, ty3) @ M_T3_center

out_w3 = int(np.ceil(max_x3 - min_x3))
out_h3 = int(np.ceil(max_y3 - min_y3))

img_T3 = apply_affine(img_T2, M_T3, (out_w3, out_h3))

cv.imshow("T3 - Rotar 90º izquierda", img_T3)
cv.waitKey(0)
cv.destroyAllWindows()

M_Tc_pre = M_T3 @ M_T2 @ M_T1

W_base, H_base = img.shape[1], img.shape[0]

min_xc, max_xc, min_yc, max_yc = transformed_bounds(W_base, H_base, M_Tc_pre)

txc = -min_xc if min_xc < 0 else 0.0
tyc = -min_yc if min_yc < 0 else 0.0

M_Tc = mat_translate(txc, tyc) @ M_Tc_pre

out_wc = int(np.ceil(max_xc - min_xc))
out_hc = int(np.ceil(max_yc - min_yc))

img_Tc = apply_affine(img, M_Tc, (out_wc, out_hc))

cv.imshow("Tc - Compuesta (T3∘T2∘T1 en un paso)", img_Tc)
cv.waitKey(0)
cv.destroyAllWindows()

# b2)


def image_fig7():
    W, H = 200, 200
    img = np.full((H, W, 3), 0, dtype=np.uint8)

    cx, cy = 100, 80
    largo = 40
    ancho = 20
    ang_deg = -60

    ang = math.radians(ang_deg)

    u = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
    u_semi = u * (largo / 2.0)

    v = np.array([-math.sin(ang), math.cos(ang)], dtype=np.float32)
    v_semi = v * (ancho / 2.0)

    C0 = np.array([cx, cy], dtype=np.float32)

    A = C0 - u_semi + v_semi
    B = C0 - u_semi - v_semi
    C = C0 + u_semi - v_semi
    D = C0 + u_semi + v_semi

    pts = np.stack([A, B, C, D]).astype(np.int32).reshape((-1, 1, 2))

    amarillo = (0, 255, 255)
    cv.fillPoly(img, [pts], amarillo)

    return img, (A, B, C, D)


img, pts = image_fig7()
cv.imshow("ROI inicial exacto (Fig7)", img)
cv.waitKey(0)
cv.destroyAllWindows()

img, (A, B, C, D) = image_fig7()
h, w = img.shape[:2]

T1 = mat_translate(-100, -80)

R = mat_rotate(30)

S = mat_scale(0.5, 0.5)

T2 = mat_translate(5, 10)

M_Tc = T2 @ S @ R @ T1

out_size = (w, h)

resultado = apply_affine(img, M_Tc, out_size)

cv.imshow("ROI1 trasladado a posición objetivo", resultado)
cv.waitKey(0)
cv.destroyAllWindows()
