"""
Facial Emotion Recognition - IP Webcam version.

Streams frames from a phone running an "IP Webcam" app (Android) instead
of a local webcam, then runs facial emotion recognition on each frame.

Setup:
    1. Install the "IP Webcam" app on your Android phone.
    2. Start the server in the app and note the IP:port it shows,
       e.g. 192.168.1.34:8080
    3. Make sure your phone and computer are on the same Wi-Fi network.
    4. Update IP_WEBCAM_URL below with your phone's address.

Usage:
    python ip_camera_emotion.py

Press ESC to quit.
"""

from facial_emotion_recognition import EmotionRecognition
import urllib.request
import cv2
import numpy as np

# Update this with your own phone's IP Webcam address
IP_WEBCAM_URL = 'http://192.168.1.34:8080/shot.jpg'

er = EmotionRecognition(device='cpu')

while True:
    img_resp = urllib.request.urlopen(IP_WEBCAM_URL)
    img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_np, -1)

    frame = er.recognise_emotion(frame, return_type='BGR')
    cv2.imshow("Facial Emotion Recognition (IP Camera)", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
