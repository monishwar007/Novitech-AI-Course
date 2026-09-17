"""
main.py
--------
Color Object Tracking + Direction Detection

Workflow (see block diagram):
1. Read frame from camera
2. Pre-process image (resize -> blur -> HSV convert -> mask -> erode/dilate)
3. Find contours in the mask
4. Draw minimum enclosing circle around the largest contour
5. Find center of the contour using image moments
6. Draw circle & center point
7. Decide direction based on radius & center position:
     - radius > 250          -> "Stop"   (object too close)
     - center_x < 150        -> "Right"
     - center_x > 450        -> "Left"
     - else                  -> "Front"

Tune redLower / redUpper using colorCalibrationForHSV.py first.
Press 'q' to quit.
"""

import cv2
import imutils

# HSV range for the object color (tune with colorCalibrationForHSV.py)
redLower = (95, 49, 100)
redUpper = (154, 255, 255)

camera = cv2.VideoCapture(0)  # Cam Init (use 1 for external webcam)

while True:
    (grabbed, frame) = camera.read()  # Read the frame
    if not grabbed:
        break

    frame = imutils.resize(frame, width=1000)  # resize
    frame = cv2.flip(frame, 1)  # mirror view, more intuitive left/right

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, redLower, redUpper)  # Mask the target colour
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  # works across OpenCV versions

    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                # Draw the enclosing circle (yellow) and center dot (red)
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                print(center, radius)

                if radius > 250:
                    direction = "Stop"
                else:
                    if center[0] < 150:
                        direction = "Right"
                    elif center[0] > 450:
                        direction = "Left"
                    elif radius < 250:
                        direction = "Front"
                    else:
                        direction = "Stop"

                print(direction)
                cv2.putText(frame, direction, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
