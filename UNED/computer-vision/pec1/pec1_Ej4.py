import cv2 as cv

cap = cv.VideoCapture(0)

ret, frame = cap.read()
frame = cv.resize(frame, (640, 480))

roi = cv.selectROI("Selecciona la mano", frame)
cv.destroyWindow("Selecciona la mano")

x, y, w, h = roi
track_window = (x, y, w, h)

roi_img = frame[y:y+h, x:x+w]
hsv_roi = cv.cvtColor(roi_img, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, (0, 30, 30), (180, 255, 255))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (640, 480))

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    backproj = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv.meanShift(backproj, track_window, criteria)

    x, y, w, h = track_window
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow("Seguimiento con MeanShift", frame)
    cv.imshow("Backprojection(en lo k se basa meanshift que va a seguir)", backproj)

    if cv.waitKey(30) & 0xFF == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
