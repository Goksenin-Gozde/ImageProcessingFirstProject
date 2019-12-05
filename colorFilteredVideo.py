import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)

    while (1):

        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([30, 40, 60])
        upper_green = np.array([100, 200, 150])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('Frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    SystemExit()
