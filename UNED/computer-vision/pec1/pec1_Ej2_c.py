import cv2 as cv
import numpy as np

# C1)
img = cv.imread("data/zigzag.jpg", cv.IMREAD_GRAYSCALE)

H, W = img.shape

mid = W // 2
left_half = img[:, :mid]
right_half = img[:, mid:]

w_left = W // 3
w_right = W - w_left

left_resized = cv.resize(left_half,  (w_left,  H),
                         interpolation=cv.INTER_LINEAR)
right_resized = cv.resize(right_half, (w_right, H),
                          interpolation=cv.INTER_LINEAR)

out = np.zeros_like(img)

out[:, :w_left] = left_resized

out[:, w_left:] = right_resized

cv.imshow("Original zigzag", img)
cv.imshow("Deformada (mitad izq -> 1/3, dcha -> 2/3)", out)
cv.waitKey(0)
cv.destroyAllWindows()

out_affine = np.zeros_like(img)

src_left = np.float32([
    [0,   0],
    [mid, 0],
    [0,   H]
])

dst_left = np.float32([
    [0,        0],
    [w_left,   0],
    [0,        H]
])

M_left = cv.getAffineTransform(src_left, dst_left)

warp_left = cv.warpAffine(
    img,
    M_left,
    (W, H),
    flags=cv.INTER_LINEAR,
    borderMode=cv.BORDER_CONSTANT,
    borderValue=0
)

out_affine[:, :w_left] = warp_left[:, :w_left]

src_right = np.float32([
    [mid, 0],
    [W,   0],
    [mid, H]
])

dst_right = np.float32([
    [w_left, 0],
    [W,      0],
    [w_left, H]
])

M_right = cv.getAffineTransform(src_right, dst_right)

warp_right = cv.warpAffine(
    img,
    M_right,
    (W, H),
    flags=cv.INTER_LINEAR,
    borderMode=cv.BORDER_CONSTANT,
    borderValue=0
)

out_affine[:, w_left:] = warp_right[:, w_left:]

cv.imshow("Original zigzag", img)
cv.imshow("Deformada (resize directo)", out_affine)
cv.waitKey(0)
cv.destroyAllWindows()

# C2)
board_path = "data/tableroAjedrez.png"
img = cv.imread(board_path, cv.IMREAD_GRAYSCALE)

H, W = img.shape

x_norm = np.linspace(0.0, 1.0, W, endpoint=False)
y_norm = np.linspace(0.0, 1.0, H, endpoint=False)
x_d, y_d = np.meshgrid(x_norm, y_norm)

x_s_norm = x_d
f = 0.4 * (x_s_norm - 0.5) ** 2 - 0.1
y_s_norm = y_d - f

x_s = x_s_norm * (W - 1)
y_s = y_s_norm * (H - 1)

x_s = np.clip(x_s, 0, W - 1).astype(np.float32)
y_s = np.clip(y_s, 0, H - 1).astype(np.float32)

deformed = cv.remap(
    img,
    x_s,
    y_s,
    interpolation=cv.INTER_LINEAR,
    borderMode=cv.BORDER_CONSTANT,
    borderValue=255
)

cv.imshow("Tablero original", img)
cv.imshow("Tablero deformado (y' = y + 0.4(x-0.5)^2 - 0.1)", deformed)
cv.waitKey(0)
cv.destroyAllWindows()
