# Novitech AI Course

A collection of hands-on Computer Vision / OpenCV mini-projects built as
part of the Novitech AI course. Each project lives in its own folder
with its own detailed README, setup instructions, and source code.

## Projects

| Project | Description |
|---|---|
| [`color-object-tracking/`](./color-object-tracking) | Tracks a colored object via webcam using HSV masking + contour detection, and reports its direction (Left / Right / Front / Stop). Includes a GUI HSV calibration tool. |
| [`face-detection/`](./face-detection) | Detects faces in an image, webcam feed, or video file using OpenCV's Haar Cascade classifier, drawing a rectangle around every detected face. |
| [`face-recognition-v2/`](./face-recognition-v2) | Trains a FisherFace recognizer on a small custom face dataset and runs live webcam recognition, labeling each detected face with a name and confidence score (or "Unknown"). Includes a synthetic test-dataset generator so the pipeline can be verified without a webcam. |
| [`facial-emotion-recognition/`](./facial-emotion-recognition) | Detects faces and classifies their emotion (happy, sad, angry, surprised, etc.) in real time from a webcam or phone IP-camera feed, using the `facial-emotion-recognition` PyTorch-based package. |

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
├── face-recognition-v2/
│   ├── create_data.py
│   ├── face_recognize.py
│   ├── generate_test_dataset.py
│   ├── download_haarcascade.py
│   └── README.md
├── facial-emotion-recognition/
│   ├── webcam_emotion.py
│   ├── ip_camera_emotion.py
│   ├── requirements.txt
│   └── README.md
└── README.md   <- you are here
```

## Author

Monishwar
[GitHub](https://github.com/monishwar007) · [LinkedIn](https://www.linkedin.com/in/monishwar-p-9a511b341)
