import cv2
from object_detector import *
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

detector = HomogeneousBgDetector()

# img = cv2.imread("phone_aruco_marker.jpg")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    _, img = cap.read()
    corners, _, _ = aruco_detector.detectMarkers(img)
    if corners:
        int_corners = np.intp(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)
        aruco_perimeter = cv2.arcLength(int_corners[0], True)

        #aruco marker being used in img has perimeter of 20cm
        pixel_cm_ratio = aruco_perimeter/20

        contours = detector.detect_objects(img)

        for cnt in contours:

            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            object_width = w/pixel_cm_ratio
            object_height = h/pixel_cm_ratio
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(img, "Width {}cm".format(round(object_width, 1)), (int(x - 100), int(y - 15)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            cv2.putText(img, "Height {}cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        # 27 is the escape key
        break
cap.release()
cv2.destroyAllWindows()