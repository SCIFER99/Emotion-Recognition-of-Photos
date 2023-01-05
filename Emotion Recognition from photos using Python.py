# By: Tim Tarver

# Emotion Recognition using Python for Machine Learning

import cv2
from deepface import DeepFace
import numpy

# Set the image path and read the image

image_path = 'face_image.jfif'
image = cv2.imread(image_path)
analyze_face = DeepFace.analyze(image, actions=['emotion'])

# Print the outcome

print(analyze_face)


