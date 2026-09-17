# Novitech AI Course

A collection of hands-on Computer Vision / OpenCV mini-projects built as
part of the Novitech AI course. Each project lives in its own folder
with its own detailed README, setup instructions, and source code.

## Projects

| Project | Description |
|---|---|
| [`color-object-tracking/`](./color-object-tracking) | Tracks a colored object via webcam using HSV masking + contour detection, and reports its direction (Left / Right / Front / Stop). Includes a GUI HSV calibration tool. |
| [`face-detection/`](./face-detection) | Detects faces in an image, webcam feed, or video file using OpenCV's Haar Cascade classifier, drawing a rectangle around every detected face. |

Each folder's own README covers:
- What the project does and how it works (with a block diagram / workflow)
- Setup and installation steps
- How to run it
- Notes and tuning tips

## General Requirements

All projects use Python 3 and OpenCV. Each project's own README lists
its exact dependencies, but at minimum you'll need:

```bash
pip install opencv-python
```

## Structure

```
Novitech-AI-Course/
├── color-object-tracking/
│   ├── main.py
│   ├── colorCalibrationForHSV.py
│   ├── colorCalibrationForHSV_gui_v2.py
│   └── README.md
├── face-detection/
│   ├── face_detection_image.py
│   ├── face_detection_webcam.py
│   ├── face_detection_video.py
│   ├── download_haarcascade.py
│   └── README.md
└── README.md   <- you are here
```

## Author

Monishwar
[GitHub](https://github.com/monishwar007) · [LinkedIn](https://www.linkedin.com/in/monishwar-p-9a511b341)
