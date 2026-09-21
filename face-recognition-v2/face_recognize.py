"""
face_recognize.py
--------------------
Trains a FisherFaceRecognizer on everything in datasets/ and then
runs live face recognition from the webcam.

Workflow (Block Diagram):
1. Loading face detection algorithm (Haar Cascade)
2. Loading classifier for face recognition (FisherFace)
3. Training classifier for our dataset (datasets/<person_name>/*.png)
4. Reading frame from camera & pre-processing (grayscale)
5. Face detection by its algorithm
6. Predicting face by loading frame into model
7. Displays recognized class with its accuracy (or "Unknown")

If a face stays "Unknown" for more than 100 consecutive frames, the
current frame is saved as input.jpg for inspection.

Press ESC to quit.
"""

import cv2
import numpy
import os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

print('Training...')
(images, labels, names, id) = ([], [], {}, 0)  # e.g. Elon : 0, Ramesh : 1

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
            # print(labels)
        id += 1

(width, height) = (130, 100)

(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# print(images, labels)
# model = cv2.face.LBPHFaceRecognizer_create()

model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)

webcam = cv2.VideoCapture(0)  # camera init (use 1 for external webcam)
cnt = 0

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # converting gray scale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 800:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]),
                        (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("input.jpg", im)
                cnt = 0

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
