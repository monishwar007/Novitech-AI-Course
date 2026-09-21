# Face Recognition (OpenCV + FisherFace)

Trains a face recognizer on a small custom dataset (one folder per
person) and then runs live webcam recognition: draws a green box
around each detected face and labels it with the person's name and
a confidence score, or "Unknown" if no trained person matches well
enough.

## How it works (Block Diagram)

1. Loading face detection algorithm (Haar Cascade)
2. Loading classifier for face recognition (`FisherFaceRecognizer`)
3. Training classifier for our dataset (`datasets/<name>/*.png`)
4. Reading frame from camera & pre-processing (grayscale)
5. Face detection by its algorithm
6. Predicting face by loading the cropped face into the trained model
7. Displays the recognized class with its confidence score (or
   "Unknown" if the match is too weak)

## Files

- `create_data.py` — captures 50 face images of ONE person from your
  webcam and saves them to `datasets/<name>/1.png` … `50.png`. Run
  this once per person, changing `sub_data = 'Ramesh'` to that
  person's name each time.
- `face_recognize.py` — walks `datasets/`, trains a
  `FisherFaceRecognizer` on every person's folder, then opens your
  webcam and recognizes faces live.
- `generate_test_dataset.py` — builds a synthetic **test** dataset
  (see below) so you can verify the whole pipeline runs correctly
  before recording your own real faces.
- `haarcascade_frontalface_default.xml` — OpenCV's pre-trained Haar
  Cascade for frontal face detection (used for both scripts above).
- `download_haarcascade.py` — fetches the Haar Cascade XML file
  automatically (it isn't checked into this repo).

## Setup

```bash
pip install opencv-contrib-python numpy
python download_haarcascade.py
```

**Important:** this project needs `opencv-contrib-python`, not plain
`opencv-python` — `cv2.face.FisherFaceRecognizer_create()` only
exists in the contrib build. If you already have plain
`opencv-python` installed, uninstall it first to avoid a conflict:

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

## Option A — Use your own webcam data (recommended for real use)

```bash
# 1. Capture 50 images of yourself (repeat per person, editing sub_data each time)
python create_data.py

# 2. Train + run live recognition
python face_recognize.py
```

Edit `sub_data = 'Ramesh'` in `create_data.py` before each run to
label each new person. Run it once per person you want recognized.

## Option B — Use the included synthetic test dataset

No webcam needed to verify the pipeline works:

```bash
python generate_test_dataset.py
python face_recognize.py   # still needs a webcam for the live loop
```

This builds two classes under `datasets/`:
- **`Lena`** — from the classic Lena test photo that ships with
  OpenCV's own sample data (`opencv/samples/data/lena.jpg`), a
  standard public-domain-use image used everywhere in computer
  vision teaching/testing.
- **`Synthetic`** — a simple cartoon face drawn with OpenCV shapes
  (not a real person), used purely to give the recognizer a second,
  clearly different class.

Each is expanded into 50 lightly augmented (rotated / brightness-
jittered / shifted) copies, matching exactly the image format
`create_data.py` produces (grayscale, 130×100 PNGs), so
`face_recognize.py` trains on them exactly as it would on real
webcam captures.

This was verified end-to-end in isolation (train + predict, without
the webcam loop): the model correctly told the two classes apart.

## Notes & tuning

- **Confidence threshold**: `if prediction[1] < 800` decides
  known-vs-unknown. FisherFace confidence is a *distance* (lower =
  more confident), so smaller numbers mean a tighter match. Tune 800
  up or down depending on how strict you want recognition to be.
  With only 2 tiny synthetic classes the model can be overconfident
  on totally unseen faces — this sharpens up with more real people
  and more varied real photos in the dataset (different angles,
  lighting, expressions).
- **`detectMultiScale` parameters** (`1.3, 5` in `face_recognize.py`,
  `1.3, 4` in `create_data.py`) control detection strictness — see
  the face-detection project's README in this repo for what
  `scaleFactor` / `minNeighbors` do.
- If your webcam is external/USB, change `cv2.VideoCapture(0)` to
  `cv2.VideoCapture(1)` in both scripts.
- For best real-world accuracy: capture each person's 50 images
  under varied lighting/expression/angle, and use at least 2–3
  people so the model has more than one class to discriminate
  between.
- "Unknown" detection saves a snapshot to `input.jpg` after 100
  consecutive low-confidence frames — useful for logging unrecognized
  visitors.

## Sample images

- `sample_photos/lena.jpg` is needed for `generate_test_dataset.py`
  and is not checked into this repo. Download it yourself:

```bash
mkdir -p sample_photos
curl -o sample_photos/lena.jpg https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg
```

## Relation to `face-recognition/`

This is a separate, independent implementation from the
`face-recognition/` folder in this repo — different test-dataset
strategy (Lena + a drawn synthetic face, vs. two real sample photos),
different augmentation approach, and this version fetches the Haar
Cascade file on demand via `download_haarcascade.py` instead of
checking in the ~900KB XML file directly.
