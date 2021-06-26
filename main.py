
import cv2
import sys
import functions
from functions import camThread

# GET classifier
classifier = functions.getClassifier()

# Open camera
camera_ids = [0, 1]

for i in camera_ids:
    camera_ids = functions.camThread("Camera" + str(i), i, classifier, 1.2, 6)
    camera_ids.start()