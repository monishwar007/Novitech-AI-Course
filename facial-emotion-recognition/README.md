# Facial Emotion Recognition

Real-time facial emotion detection using the [`facial-emotion-recognition`](https://pypi.org/project/facial-emotion-recognition/)
Python package (built on PyTorch + OpenCV). It detects faces in a video
stream and classifies the emotion on each face (happy, sad, angry,
surprised, neutral, etc.), drawing the result directly on the frame.

Two capture modes are included:

- **`webcam_emotion.py`** — uses your computer's built-in/USB webcam.
- **`ip_camera_emotion.py`** — streams frames from a phone running the
  Android **IP Webcam** app, useful if your laptop has no camera or you
  want a better camera angle.

## How it works

1. `EmotionRecognition(device='cpu')` loads a pretrained face-detection +
   emotion-classification model.
2. Each video frame is passed to `er.recognise_emotion(frame, return_type='BGR')`,
   which finds faces, predicts an emotion for each, and draws bounding
   boxes + labels on the frame.
3. The annotated frame is displayed with `cv2.imshow` in a loop until you
   press **ESC**.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (CPU-only machines) Patch PyTorch's `torch.load` default

The pretrained model weights bundled with this package were saved from a
GPU session. On a machine with no CUDA GPU, loading them can raise a
deserialization error unless `torch.load` is told to map tensors to the
CPU by default.

Find this file in your environment:

```
<your-venv>/lib/pythonX.X/site-packages/torch/serialization.py
```

Change:

```python
def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
```

to:

```python
def load(f, map_location='cpu', pickle_module=pickle, **pickle_load_args):
```

> You only need this if you actually hit a CUDA-related error when
> running the scripts below on a CPU-only machine.

### 3. Run it

**Webcam:**

```bash
python webcam_emotion.py
```

**Phone camera (IP Webcam app):**

1. Install "IP Webcam" from the Play Store on your Android phone.
2. Start the server in the app; it will show an address like
   `192.168.1.34:8080`.
3. Edit `IP_WEBCAM_URL` in `ip_camera_emotion.py` to match your phone's
   address.
4. Run:

```bash
python ip_camera_emotion.py
```

Press **ESC** in the video window to quit either script.

## Project structure

```
facial-emotion-recognition/
├── webcam_emotion.py       # Local webcam capture + emotion recognition
├── ip_camera_emotion.py    # Phone (IP Webcam) capture + emotion recognition
├── requirements.txt
└── README.md
```

## Notes

- Works best with good, even lighting and the face reasonably centered.
- Multiple faces in frame are all detected and labeled independently.
- To use a GPU instead of CPU, change `device='cpu'` to `device='gpu'`
  in either script (requires a CUDA-capable GPU and PyTorch with CUDA
  support installed).
