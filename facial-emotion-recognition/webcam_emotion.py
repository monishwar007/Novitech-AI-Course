"""
Facial Emotion Recognition - Local Webcam version.

Detects faces from your computer's webcam feed in real time and
overlays the predicted emotion (happy, sad, angry, surprised, etc.)
on the video frame.

Usage:
    python webcam_emotion.py

Press ESC to quit.
"""

from facial_emotion_recognition import EmotionRecognition
import cv2

# Change device='cpu' to device='gpu' if you have a CUDA-enabled GPU
er = EmotionRecognition(device='cpu')

# 0 = default webcam. Change to 1, 2, ... if you have multiple cameras.
cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    frame = er.recognise_emotion(frame, return_type='BGR')
    cv2.imshow("Facial Emotion Recognition", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
