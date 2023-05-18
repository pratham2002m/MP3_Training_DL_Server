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


# calculate a face embedding for each face in the dataset using facenet
def get_embedding(model, face_pixels):
        """Get the face embedding for one face"""
        # scale pixel values
        print(face_pixels.shape)
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        
        # make prediction to get embedding
        yhat = model.predict(samples)
        # print(yhat)
        
        return yhat[0]

def face_to_embedings(faces, model):
        """Convert each face in the train set to an embedding."""
        embedings = []
        for face_pixels in faces:
            embedding = get_embedding(model, face_pixels)
            embedings.append(embedding)
        embedings = asarray(embedings)
        return embedings


train = load('face_train_dataset.npz')
test = load('face_test_dataset.npz')

train_X, train_Y, test_X, test_Y = train['arr_0'], train['arr_1'], test['arr_0'], test['arr_1']

model = load_model('C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/Final Repo/facenet_keras.h5', compile=False)

print(train_X.shape)

newTrainX = face_to_embedings(train_X, model)
newTestX = face_to_embedings(test_X, model)
savez_compressed('C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/Final Repo/face_train_embeddings.npz', newTrainX, train_Y)
savez_compressed('C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/Final Repo/face_test_embeddings.npz',  newTestX, test_Y)


print('Loaded: ', train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
print('Loaded Model')
print(newTrainX.shape)
print(newTestX.shape)


train = load('C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/Final Repo/face_train_embeddings.npz')
test = load('C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/Final Repo/face_test_embeddings.npz')
train_X, train_Y, test_X, test_Y = train['arr_0'], train['arr_1'], test['arr_0'], test['arr_1']
print(f'Dataset: train={train_X.shape[0]}, test={test_X.shape[0]}')

# normalize input vectors
in_encoder = Normalizer(norm='l2')

# print(train_X)
# print(test_X)
train_X = in_encoder.transform(train_X)
test_X = in_encoder.transform(test_X)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(train_Y)
train_Y = out_encoder.transform(train_Y)
test_Y = out_encoder.transform(test_Y)
model = SVC(kernel='linear', probability=True)
model.fit(train_X, train_Y)
yhat_train = model.predict(train_X)
yhat_test = model.predict(test_X)
score_train = accuracy_score(train_Y, yhat_train)
score_test = accuracy_score(test_Y, yhat_test)

print(f'Accuracy: train={score_train * 100:.3f}, test={score_test * 100:.3f}')