"""
face_detection_image.py
-------------------------
Face Detection on a static image using OpenCV's Haar Cascade classifier.

Workflow:
1. Load the Haar Cascade face detection algorithm (XML)
2. Read the image
3. Convert the color image to grayscale
4. Run detectMultiScale() to get face coordinates
5. Draw a rectangle on each detected face
6. Display the output

Usage:
    python face_detection_image.py
    (make sure 'crick.jpg' -- or whatever image you want -- is in the
     same folder, and haarcascade_frontalface_default.xml is present)
"""

import cv2

# Importing OpenCV package -- already done above

# Reading the image
img = cv2.imread('crick.jpg')

if img is None:
    raise FileNotFoundError(
        "Could not read 'crick.jpg'. Place an image with that name "
        "in this folder, or change the filename in this script."
    )

# Converting image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Loading the required haar-cascade xml classifier file
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Applying the face detection method on the grayscale image
# faces = face_cascade.detectMultiScale(src, scaleFactor, minNeighbors)
#   scaleFactor  -> how much the image size is reduced at each image scale
#   minNeighbors -> how many neighbors each candidate rectangle should
#                   have to retain it
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

print(f"Found {len(faces_rect)} face(s)")

# Iterating through rectangles of detected faces
for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Detected faces', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
