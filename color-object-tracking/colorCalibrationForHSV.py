"""
colorCalibrationForHSV.py
--------------------------
Interactive HSV calibration tool.

Move the sliders until ONLY your target colored object shows up
white in the "Mask" window (everything else black).
Then note down the Low/High values for H, S, V — these become
your redLower / redUpper (or blueLower/blueUpper etc.) tuples
in main.py.

Press 'q' to quit.
"""

import cv2
import numpy as np


def nothing(x):
    pass


# Use 0 for built-in webcam, 1 for an external/USB webcam
camera = cv2.VideoCapture(0)

cv2.namedWindow("Sliders")
cv2.resizeWindow("Sliders", 400, 400)

# Create Hue, Saturation, Value trackbars (Low & High for each)
cv2.createTrackbar("Hue Low", "Sliders", 0, 179, nothing)
cv2.createTrackbar("Hue High", "Sliders", 179, 179, nothing)
cv2.createTrackbar("Sat Low", "Sliders", 0, 255, nothing)
cv2.createTrackbar("Sat High", "Sliders", 255, 255, nothing)
cv2.createTrackbar("Val Low", "Sliders", 0, 255, nothing)
cv2.createTrackbar("Val High", "Sliders", 255, 255, nothing)

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break

    frame = cv2.resize(frame, (500, 375))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hL = cv2.getTrackbarPos("Hue Low", "Sliders")
    hH = cv2.getTrackbarPos("Hue High", "Sliders")
    sL = cv2.getTrackbarPos("Sat Low", "Sliders")
    sH = cv2.getTrackbarPos("Sat High", "Sliders")
    vL = cv2.getTrackbarPos("Val Low", "Sliders")
    vH = cv2.getTrackbarPos("Val High", "Sliders")

    lower = np.array([hL, sL, vL])
    upper = np.array([hH, sH, vH])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print(f"redLower = ({hL}, {sL}, {vL})")
        print(f"redUpper = ({hH}, {sH}, {vH})")
        break

camera.release()
cv2.destroyAllWindows()
