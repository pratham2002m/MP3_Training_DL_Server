from keras.models import load_model
import keras


import os
import numpy as np
import cv2

from PIL import Image
from numpy import asarray, savez_compressed, expand_dims, load
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isdir
from keras.models import load_model
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot 
from sklearn.metrics import accuracy_score

import cv2
import os


def merge_image(back, front, x, y):
    # convert to rgba
    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

    # crop the overlay from both images
    bh, bw = back.shape[:2]
    fh, fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x + fw, bw)
    y1, y2 = max(y, 0), min(y + fh, bh)
    front_cropped = front[y1 - y:y2 - y, x1 - x:x2 - x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:, :, 3:4] / 255
    alpha_back = back_cropped[:, :, 3:4] / 255

    # replace an area in result with overlay
    result = back.copy()
    return result


def make_smaller(img, scale_percent=15):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    dsize = (width, height)

    # resize image
    small_image = cv2.resize(img, dsize)
    return small_image


def augment_selfie(img, backg_img):
    img = make_smaller(img)
    res = merge_image(backg_img, img, 150, 150)
    return res


def create_augmented_image(path, background):
    folder, filename = os.path.split(path)
    res = augment_selfie(cv2.imread(path), background)
    os.chdir(folder)
    name, extension = os.path.splitext(filename)
    new_filename = f"{name}_aug{extension}"
    cv2.imwrite(new_filename, res)


def rewrite_to_augmented(path, background):
    folder, filename = os.path.split(path)
    res = augment_selfie(cv2.imread(path), background)
    os.chdir(folder)
    cv2.imwrite(filename, res)


def extract_face(filename, required_size=(160, 160)):
        """
        Extract a single face from a given photograph
        """
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        if len(results) == 0:
            return
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        return asarray(image)

folder = 'C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/Final Repo/dataset/train/pratham'
i = 1
# # # enumerate files
print(listdir(folder))
for filename in listdir(folder):
    path = os.path.join(folder, filename)
    face = extract_face(path)
    print(path)
    if face is None:
      print(f'I can`t find a person in {filename}!/nI will try to use augmentation./n')
      back = cv2.imread('backg.jpg')
      rewrite_to_augmented(path, back)
      continue
    print(i, face.shape)
    # plot
    pyplot.subplot(4, 7, i)
    pyplot.axis('off')
    pyplot.imshow(face)
    i += 1
pyplot.show()