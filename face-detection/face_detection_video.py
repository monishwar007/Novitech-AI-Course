"""
face_detection_video.py
-------------------------
Face Detection on a video file (or webcam) using OpenCV's Haar Cascade
classifier.

Set video_path to a video file to run detection on that file, or to 0
to use your default webcam instead.

Press ESC (or 'q') to quit.
"""

import cv2

alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Replace 'path/to/your/video.mp4' with the path to your video file,
# or set this to 0 to use the webcam instead.
video_path = "check.mp4"
cam = cv2.VideoCapture(video_path)

if not cam.isOpened():
    raise FileNotFoundError(
        f"Could not open video source '{video_path}'. "
        "Check the path, or set video_path = 0 to use your webcam."
    )

while True:
    ret, img = cam.read()

    if not ret:
        print("Error reading video file (or reached the end). Exiting...")
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (155, 155, 255), 2)

    cv2.imshow("FaceDetection", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
