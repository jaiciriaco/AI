import cv2 as cv
import numpy as np
import glob
import os

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

pattern_size = (3, 7)

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob("data/calib_movil/*.jpeg")

for fname in images:
    img = cv.imread(fname)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    if ret:
        corners2 = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv.imshow('Esquinas móvil', img)
        cv.waitKey(200)

cv.destroyAllWindows()

if len(objpoints) > 0:
    img_size = gray.shape[::-1]

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    print("\nResultados de la calibración:")
    print("Error medio de reproyección:", ret)
    print("\nMatriz intrínseca (K_movil):\n", mtx)
    print("\nCoeficientes de distorsión (dist_movil):\n", dist.ravel())

    np.savez('B_movil.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("\nParámetros de calibración guardados en 'B_movil.npz'")

    img = cv.imread(images[0])
    h, w = img.shape[:2]

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w_roi, h_roi = roi
    dst_cropped = dst[y:y + h_roi, x:x + w_roi]

    os.makedirs("resultados", exist_ok=True)

    cv.imshow('Original móvil', img)
    cv.imshow('Móvil undistort', dst_cropped)
    cv.waitKey(0)
    cv.destroyAllWindows()
