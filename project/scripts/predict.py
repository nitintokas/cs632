import sys
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
from keras.models import load_model
import cv2
from PIL import Image
import glob

model = load_model("/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/scripts/final_model_now.h5")
cam = cv2.VideoCapture(0)

cam.set(3,150);
cam.set(4,150);
s, im = cam.read() # captures image

im2 = cv2.resize(im, (64, 64))
im2.shape

im2 = im2.reshape(3,64,64)
a = np.asarray(im2)

arr2=[]
arr2.append(a)
cam.release()
arr =np.array(arr2)

score = model.predict(arr, batch_size=24)
print(score)