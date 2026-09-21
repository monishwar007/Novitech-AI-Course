"""
generate_test_dataset.py
--------------------------
Builds a small TEST dataset so you can verify create_data.py's output
format and face_recognize.py's training/recognition pipeline end-to-
end WITHOUT needing a webcam.

Two "identities" are generated:
  - "Lena"     -> from the classic public-domain-use Lena test image
                  that ships with OpenCV's own sample data
                  (opencv/samples/data/lena.jpg), a standard image
                  used everywhere in computer-vision teaching.
  - "Synthetic" -> a procedurally drawn cartoon face (not a real
                  person), used purely so the recognizer has a
                  second, clearly distinguishable class to tell apart
                  from Lena.

Each identity is expanded to 50 slightly-augmented (rotated/shifted/
brightness-jittered) crops, matching the exact format create_data.py
produces: datasets/<name>/1.png ... 50.png, grayscale, 130x100.

Run this once, then run face_recognize.py directly -- no webcam
needed for this synthetic run (see the --use_webcam flag notes in
the README for the switch back to a live camera).
"""

import os
import cv2
import numpy as np

HAAR_FILE = 'haarcascade_frontalface_default.xml'
DATASETS = 'datasets'
SAMPLE_DIR = 'sample_photos'
WIDTH, HEIGHT = 130, 100
IMAGES_PER_CLASS = 50

face_cascade = cv2.CascadeClassifier(HAAR_FILE)


def augment(gray_face, rng):
    """Small random rotation + brightness jitter + slight resize jitter
    so 50 'different' training images can be produced from one base
    photo (stands in for 50 separate webcam captures)."""
    h, w = gray_face.shape

    angle = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray_face, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    brightness = rng.uniform(0.85, 1.15)
    jittered = np.clip(rotated.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    dx, dy = rng.integers(-3, 4), rng.integers(-3, 4)
    M2 = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(jittered, M2, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return shifted


def save_class(name, base_gray_face):
    path = os.path.join(DATASETS, name)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(hash(name) % (2**32))
    resized = cv2.resize(base_gray_face, (WIDTH, HEIGHT))
    for i in range(1, IMAGES_PER_CLASS + 1):
        out = augment(resized, rng)
        cv2.imwrite(os.path.join(path, f'{i}.png'), out)
    print(f"Saved {IMAGES_PER_CLASS} images to {path}/")


def build_lena_class():
    lena_path = os.path.join(SAMPLE_DIR, 'lena.jpg')
    img = cv2.imread(lena_path)
    if img is None:
        raise FileNotFoundError(
            f"'{lena_path}' not found. Download it from "
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg "
            f"and place it in {SAMPLE_DIR}/"
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    if len(faces) == 0:
        raise RuntimeError("No face detected in lena.jpg -- cascade/image mismatch.")
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y + h, x:x + w]
    save_class('Lena', face)


def build_synthetic_class():
    """Draws a simple cartoon face so the recognizer has a second,
    clearly different class. This is NOT a real person's photo."""
    canvas = np.full((300, 300), 220, dtype=np.uint8)  # light gray background
    center = (150, 150)
    cv2.circle(canvas, center, 110, 40, -1)             # face (dark circle)
    cv2.circle(canvas, (110, 120), 15, 220, -1)          # left eye
    cv2.circle(canvas, (190, 120), 15, 220, -1)          # right eye
    cv2.ellipse(canvas, (150, 190), (40, 20), 0, 0, 180, 220, 6)  # smile
    cv2.rectangle(canvas, (60, 60), (240, 100), 20, -1)  # hair band

    faces = face_cascade.detectMultiScale(canvas, 1.1, 3)
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face = canvas[y:y + h, x:x + w]
    else:
        # Haar cascades won't reliably detect a drawn cartoon face --
        # that's expected, so fall back to the full canvas as the
        # "face" region for this synthetic class.
        face = canvas

    save_class('Synthetic', face)


if __name__ == '__main__':
    os.makedirs(DATASETS, exist_ok=True)
    build_lena_class()
    build_synthetic_class()
    print("\nTest dataset ready under datasets/. You can now run:")
    print("    python face_recognize.py")
