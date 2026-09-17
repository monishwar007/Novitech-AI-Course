"""
download_haarcascade.py
-------------------------
Downloads OpenCV's official pre-trained Haar Cascade frontal-face
classifier XML into this folder.

The file is ~930 KB, so instead of committing it as a binary blob into
the course repo, this script fetches it straight from OpenCV's GitHub
whenever you need it.

Run this ONCE before using face_detection_image.py,
face_detection_webcam.py, or face_detection_video.py:

    python download_haarcascade.py
"""

import os
import urllib.request

URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/"
    "haarcascades/haarcascade_frontalface_default.xml"
)
OUT_FILE = "haarcascade_frontalface_default.xml"

if os.path.exists(OUT_FILE):
    print(f"'{OUT_FILE}' already exists — nothing to do.")
else:
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, OUT_FILE)
    print(f"Saved to '{OUT_FILE}'.")
