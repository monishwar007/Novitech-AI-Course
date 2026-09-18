"""
generate_test_dataset.py
--------------------------
Builds a small TEST dataset for face_recognize.py using real, freely
licensed sample face photos (shipped with OpenCV's own sample/test
data), instead of requiring a live webcam.

For each of 2 "identities" (Lena, Audrey), it:
1. Runs the same Haar Cascade detector used in create_data.py
2. Crops the detected face
3. Applies small random augmentations (rotation, brightness, flip) to
   turn ONE source photo into ~20 slightly-varied samples -- similar
   to how create_data.py captures 50 slightly different webcam frames
   of the same person
4. Resizes each to (130, 100) grayscale and saves into
   datasets/<Name>/1.png ... N.png -- the exact format
   face_recognize.py expects

This lets you test-run face_recognize.py end-to-end without a webcam.
For your real project, replace this dataset by running
create_data.py against your own webcam for each person.
"""

import cv2
import numpy as np
import os

SOURCES = {
    "Lena": "sample_photos/lena.jpg",
    "Audrey": "sample_photos/audrybt1.png",
}

HAAR_FILE = "haarcascade_frontalface_default.xml"
DATASETS = "datasets"
WIDTH, HEIGHT = 130, 100
SAMPLES_PER_PERSON = 20

face_cascade = cv2.CascadeClassifier(HAAR_FILE)


def augment(face_gray, rng):
    """Small random rotation + brightness jitter to simulate
    natural frame-to-frame variation from a webcam capture."""
    h, w = face_gray.shape
    angle = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(face_gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    brightness = rng.randint(-15, 16)
    adjusted = np.clip(rotated.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return adjusted


def main():
    os.makedirs(DATASETS, exist_ok=True)
    rng = np.random.RandomState(7)

    for name, src_path in SOURCES.items():
        img = cv2.imread(src_path)
        if img is None:
            print(f"Could not read {src_path}, skipping {name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        if len(faces) == 0:
            print(f"No face detected in {src_path}, skipping {name}")
            continue

        # take the largest detected face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        base_face = gray[y:y + h, x:x + w]
        base_resized = cv2.resize(base_face, (WIDTH, HEIGHT))

        out_dir = os.path.join(DATASETS, name)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(1, SAMPLES_PER_PERSON + 1):
            sample = augment(base_resized, rng)
            cv2.imwrite(os.path.join(out_dir, f"{i}.png"), sample)

        print(f"{name}: saved {SAMPLES_PER_PERSON} samples to {out_dir}")

    print("Test dataset ready.")


if __name__ == "__main__":
    main()
