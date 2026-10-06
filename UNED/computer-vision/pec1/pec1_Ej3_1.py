import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob("data/calib/*.jpg")

for fname in images:
    img = cv.imread(fname)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(200)

cv.destroyAllWindows()

if len(objpoints) > 0:
    img_size = gray.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    print("Error medio de reproyección:", ret)
    print("Matriz intrínseca (K):\n", mtx)
    print("Coeficientes de distorsión:\n", dist.ravel())

    img = cv.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w_roi, h_roi = roi
    dst = dst[y:y+h_roi, x:x+w_roi]

    mapx, mapy = cv.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), cv.CV_32FC1
    )
    dst2 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    dst2 = dst2[y:y+h_roi, x:x+w_roi]

    axis = np.float32([
        [0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
        [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]
    ])

    def draw_cube(img_c, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)

        img_c = cv.drawContours(img_c, [imgpts[:4]], -1, (0, 255, 255), -3)

        for i, j in zip(range(4), range(4, 8)):
            img_c = cv.line(img_c, tuple(imgpts[i]), tuple(imgpts[j]),
                            (255, 0, 0), 3)

        img_c = cv.drawContours(img_c, [imgpts[4:]], -1, (0, 255, 0), 3)

        return img_c

    test_img_path = images[12]
    print("Usando para el cubo:", test_img_path)

    img_cubo = cv.imread(test_img_path)
    gray_cubo = cv.cvtColor(img_cubo, cv.COLOR_BGR2GRAY)

    ret_c, corners_c = cv.findChessboardCorners(gray_cubo, (7, 6), None)

    if ret_c:
        corners2_c = cv.cornerSubPix(
            gray_cubo, corners_c, (11, 11), (-1, -1), criteria
        )

        ret_pnp, rvec_c, tvec_c = cv.solvePnP(objp, corners2_c, mtx, dist)

        imgpts, _ = cv.projectPoints(axis, rvec_c, tvec_c, mtx, dist)

        img_cubo = draw_cube(img_cubo, imgpts)

        cv.imshow('Cubo sobre tablero', img_cubo)
        cv.waitKey(0)
        cv.destroyAllWindows()
