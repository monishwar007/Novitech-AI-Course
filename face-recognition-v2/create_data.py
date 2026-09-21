"""
create_data.py
----------------
Captures 50 face images of one person from the webcam and saves them
into datasets/<sub_data>/ as 1.png ... 50.png. Run this once per
person you want the recognizer to learn (change sub_data each time).

Workflow:
1. Load Haar Cascade face detector
2. Create datasets/<sub_data> folder if it doesn't exist
3. Read frames from webcam
4. Detect face, crop, resize to (130, 100), convert to grayscale
5. Save each cropped face as a numbered .png
6. Stop after 50 images (or press ESC to quit early)
"""

import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Change this for every new person you capture
sub_data = 'Ramesh'

path = os.path.join(datasets, sub_data)  # datasets/Ramesh
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)

webcam = cv2.VideoCapture(0)  # Camera init (use 1 for external webcam)

count = 1
while count < 51:
    print(count)
    (_, im) = webcam.read()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count += 1

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:  # ESC
        break

webcam.release()
cv2.destroyAllWindows()
print(f"Saved {count - 1} images to {path}/")
