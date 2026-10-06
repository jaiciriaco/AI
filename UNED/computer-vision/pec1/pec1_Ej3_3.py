import cv2 as cv
import numpy as np

with np.load('B_movil.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

print("Matriz intrínseca:\n", mtx)
print("Distorsión:\n", dist.ravel())

img = cv.imread('data/face/image.png')

scale = 0.5
img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

h, w = img.shape[:2]
image_points = []

puntos_nombre = [
    "1) Punta de la NARIZ",
    "2) MENTÓN",
    "3) Esquina EXTERNA del OJO IZQUIERDO",
    "4) Esquina EXTERNA del OJO DERECHO",
    "5) Comisura IZQUIERDA de la BOCA",
    "6) Comisura DERECHA de la BOCA"
]

print("Haz clic en la ventana en este orden:")
for txt in puntos_nombre:
    print("   ", txt)


def on_mouse(event, x, y, flags, param):
    global image_points, img

    if event == cv.EVENT_LBUTTONDOWN and len(image_points) < 6:
        image_points.append((x, y))
        cv.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv.putText(img, str(len(image_points)), (x + 5, y - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow("Marca los puntos de la cara", img)

        if len(image_points) < 6:
            print(
                f"Punto {len(image_points)} marcado. Siguiente: {puntos_nombre[len(image_points)]}")
        else:
            print("Ya se han marcado los 6 puntos.")


cv.namedWindow("Marca los puntos de la cara")
cv.setMouseCallback("Marca los puntos de la cara", on_mouse)

print("\nHaz clic en la ventana y marca los 6 puntos.")
print("Cuando termines, pulsa cualquier tecla en la ventana para continuar.")

while True:
    cv.imshow("Marca los puntos de la cara", img)
    key = cv.waitKey(10) & 0xFF
    if key != 255 and len(image_points) == 6:
        break

cv.destroyWindow("Marca los puntos de la cara")

if len(image_points) != 6:
    print("No se han marcado los 6 puntos. Saliendo.")
    exit()

image_points = np.array(image_points, dtype=np.float32)

model_points = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -110.0, -30.0],
    [-80.0, 40.0, -30.0],
    [80.0, 40.0, -30.0],
    [-60.0, -50.0, -30.0],
    [60.0, -50.0, -30.0]
], dtype=np.float32)

success, rvec, tvec = cv.solvePnP(
    model_points,
    image_points,
    mtx,
    dist,
    flags=cv.SOLVEPNP_ITERATIVE
)

print("\nVector de rotación (rvec):\n", rvec)
print("\nVector de traslación (tvec):\n", tvec)

axis = np.float32([
    [0, 0, 200],
    [200, 0, 0],
    [0, -200, 0],
])

imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

nose_point = tuple(image_points[0].astype(int))

pZ = tuple(imgpts[0].ravel().astype(int))
pX = tuple(imgpts[1].ravel().astype(int))
pY = tuple(imgpts[2].ravel().astype(int))

cv.line(img, nose_point, pZ, (0, 0, 255), 3)
cv.line(img, nose_point, pX, (255, 0, 0), 3)
cv.line(img, nose_point, pY, (0, 255, 0), 3)

cv.imshow("Resultado pose de la cabeza", img)

cv.waitKey(0)
cv.destroyAllWindows()
