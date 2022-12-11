from PIL import Image
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
from preprocessing import *
detector = MTCNN()
img = Image.open("D:\MTCNN\images\Duc\Duc0.png")
img = np.array(img)
result = detector.detect_faces(img)

x1, y1, width, height = result[0]['box']
x1, y1 = abs(x1), abs(y1)
x2, y2 = x1 + width, y1 + height
face = img[y1:y2, x1:x2]
image = Image.fromarray(face)
image = image.resize((160, 160))
face_array = np.asarray(image)
print(face_array.shape)
import os 
a = os.listdir("D:\MTCNN\images\Duc")
print(a)
print(len(a))
def load_face(dir):
    faces = []
    for filename in os.listdir(dir):
        path = dir + "\\" + filename
        img, resized_img = face_extract(path)
        faces.append(resized_img)
    return faces
face = load_face("D:\MTCNN\Train\Duc")
print(face)
print(len(face))
