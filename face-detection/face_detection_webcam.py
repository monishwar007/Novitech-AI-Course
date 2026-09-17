"""
face_detection_webcam.py
--------------------------
Real-time Face Detection using OpenCV's Haar Cascade classifier and
a live webcam feed.

Workflow (Block Diagram):
1. Loading HaarCascade Face Algorithm
2. Initializing Camera
3. Reading Frame from Camera
4. Converting Color image into Grayscale Image
5. Obtaining Face coordinates by passing algorithm
6. Drawing Rectangle on the Face Coordinates
7. Display the output

Press ESC to quit.
"""

import cv2

alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)  # Loading algorithm

cam = cv2.VideoCapture(0)  # Cam id initialization (use 1 for external webcam)

while True:
    _, img = cam.read()  # Reading the frame from cam

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converting clr image to gray

    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)  # Getting coordinates

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("FaceDetection", img)

    key = cv2.waitKey(10)
    if key == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
