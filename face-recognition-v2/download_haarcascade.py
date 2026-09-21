"""
download_haarcascade.py
-------------------------
Downloads OpenCV's official pre-trained Haar Cascade frontal-face
classifier XML into this folder.

Run this ONCE before using create_data.py or face_recognize.py:

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
