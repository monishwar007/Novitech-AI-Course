# Face Detection using Haar Cascade (OpenCV)

Detects faces in an image, a webcam feed, or a video file using OpenCV's
pre-trained Haar Cascade classifier, and draws a rectangle around every
detected face.

## How it works (Block Diagram)

1. Loading HaarCascade Face Algorithm
2. Initializing Camera (or opening image/video)
3. Reading Frame from Camera
4. Converting Color image into Grayscale Image
5. Obtaining Face coordinates by passing algorithm (`detectMultiScale`)
6. Drawing Rectangle on the Face Coordinates
7. Display the output

## `detectMultiScale`

```python
faces = face_cascade.detectMultiScale(src, scaleFactor, minNeighbors)
```

- **scaleFactor** — how much the image size is reduced at each image
  scale (e.g. `1.1` = reduce by 10% each pass; smaller = more accurate
  but slower).
- **minNeighbors** — how many neighboring candidate rectangles a
  detection needs to be retained (higher = fewer false positives, but
  may miss faces).

## Files

- `download_haarcascade.py` — run this FIRST. Downloads OpenCV's
  official pre-trained Haar Cascade XML (`haarcascade_frontalface_default.xml`,
  ~930 KB) into this folder.
- `face_detection_image.py` — detect faces in a single still image
  (`crick.jpg` by default — replace with your own image and update
  the filename in the script).
- `face_detection_webcam.py` — real-time face detection from a live
  webcam feed.
- `face_detection_video.py` — face detection on a video file
  (`check.mp4` by default — set `video_path = 0` to use your webcam
  instead).

## Setup

```bash
pip install opencv-python
python download_haarcascade.py
```

## Run

```bash
# Static image
python face_detection_image.py

# Webcam
python face_detection_webcam.py

# Video file
python face_detection_video.py
```

- Press `ESC` (webcam/video scripts) or any key (image script) to
  close the window.
- If your webcam is external/USB, change `cv2.VideoCapture(0)` to
  `cv2.VideoCapture(1)`.
- Tune `scaleFactor` / `minNeighbors` in the `detectMultiScale()` call
  if detection is too strict, too loose, or too slow.

## Notes

- The Haar Cascade XML must be in the same folder as the script (or
  give the full path to it) — it's the pre-trained model that
  `CascadeClassifier` loads. Get it by running
  `download_haarcascade.py`, or download it directly from
  [OpenCV's GitHub](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml).
- This detects **frontal faces**. For side profiles, OpenCV ships a
  separate `haarcascade_profileface.xml` cascade.
