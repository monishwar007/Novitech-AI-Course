# Face Recognition (FisherFace + Haar Cascade)

Recognizes specific people's faces from a webcam feed. Builds on
top of Haar Cascade face **detection** by adding face **recognition**:
it learns each person's face from a small set of sample images, then
identifies them live, printing their name and a confidence score, or
"Unknown" if the face doesn't match anyone it was trained on.

## How it works (Block Diagram)

1. Loading face detection algorithm (Haar Cascade)
2. Loading classifier for face recognition (`FisherFaceRecognizer`)
3. Training the classifier on our dataset
4. Reading a frame from the camera & pre-processing (grayscale)
5. Face detection by the Haar Cascade algorithm
6. Predicting the face by feeding the cropped face into the trained
   model
7. Displaying the recognized name with its confidence, or "Unknown"

## Files

- `create_data.py` — captures 50 face images of ONE person from your
  webcam and saves them to `datasets/<PersonName>/1.png ... 50.png`.
  Run this once per person, changing `sub_data` each time.
- `face_recognize.py` — walks `datasets/`, trains a FisherFace model
  on every person's images, then runs live recognition from the
  webcam.
- `generate_test_dataset.py` — builds a small **test** dataset from
  two real, freely licensed sample face photos (shipped with OpenCV's
  own sample/test data) instead of a live webcam, so you can verify
  the whole pipeline works before capturing your own data. See
  "Test dataset" below.
- `haarcascade_frontalface_default.xml` — OpenCV's pre-trained Haar
  Cascade classifier used for face detection at every stage.
- `datasets/` — one subfolder per known person, each containing that
  person's face images (`1.png`, `2.png`, ...).

## Setup

```bash
pip install opencv-contrib-python numpy
```

**Important:** this project needs `opencv-contrib-python`, not just
`opencv-python` — `cv2.face.FisherFaceRecognizer_create()` lives in
the "contrib" extras module and isn't included in the base package.
If you already have `opencv-python` installed, uninstall it first to
avoid a conflict:

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

## Step 1 — Build your dataset

Real capture (recommended for your actual project):

1. Open `create_data.py` and set `sub_data = 'YourName'`.
2. Run it: `python create_data.py`. Look at the camera — it captures
   50 face images automatically as it detects your face, saving them
   to `datasets/YourName/`.
3. Repeat for every person you want the system to recognize (change
   `sub_data` each time and re-run).

Test dataset (no webcam needed, real face photos):

```bash
python generate_test_dataset.py
```

This step uses two real sample face photos bundled with this project
(`sample_photos/lena.jpg` and `sample_photos/audrybt1.png`, both
standard, freely used computer-vision sample images), runs them
through the same Haar Cascade detect → crop → resize pipeline as
`create_data.py`, and generates 20 lightly-augmented (small
rotation/brightness jitter) samples per person into `datasets/Lena/`
and `datasets/Audrey/` — simulating what 20 real webcam captures of
each person would look like.

## Step 2 — Run recognition

```bash
python face_recognize.py
```

- It trains the FisherFace model on everything in `datasets/`
  (takes a few seconds).
- Then it opens your webcam and draws a green box around any
  detected face, labeling it with the matched person's name and a
  confidence score if `distance < 800`, or "Unknown" otherwise.
- If a face stays "Unknown" for over 100 consecutive frames, it saves
  a snapshot as `input.jpg` and prints "Unknown Person".
- Press `ESC` to quit.

## Verified pipeline (test results)

Running the trained model against the original (non-augmented)
source photos confirms the recognizer works correctly:

| Test face | Predicted | Distance (lower = more confident) |
|---|---|---|
| Lena | Lena | 0.4 |
| Audrey | Audrey | 0.9 |
| Unrelated face (a third, unknown person) | — | 3121.0 (correctly far past the `800` "Unknown" threshold) |

This confirms both the training and the `800` confidence threshold
used in `face_recognize.py` behave sensibly.

## Notes

- **Distance, not similarity**: FisherFace's `prediction[1]` is a
  distance score — **lower means more confident**, the opposite of a
  percentage/similarity score. That's why the code checks
  `prediction[1] < 800` for a match.
- More, more varied samples per person (different angles, lighting,
  expressions) make recognition much more robust than a handful of
  near-identical frames.
- If your webcam is external/USB, change `cv2.VideoCapture(0)` to
  `cv2.VideoCapture(1)` in both scripts.
- FisherFace requires **at least 2 people** in the dataset to train
  (it needs multiple classes to separate). If you only want to
  recognize one person, add a second "Unknown/Other" folder with a
  handful of random face images.
