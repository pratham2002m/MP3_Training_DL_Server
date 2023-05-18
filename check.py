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

import sqlite3


conn = sqlite3.connect(str(os.getcwd()) + '/Database.db')


# def merge_image(back, front, x, y):
#     # convert to rgba
#     if back.shape[2] == 3:
#         back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
#     if front.shape[2] == 3:
#         front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

#     # crop the overlay from both images
#     bh, bw = back.shape[:2]
#     fh, fw = front.shape[:2]
#     x1, x2 = max(x, 0), min(x + fw, bw)
#     y1, y2 = max(y, 0), min(y + fh, bh)
#     front_cropped = front[y1 - y:y2 - y, x1 - x:x2 - x]
#     back_cropped = back[y1:y2, x1:x2]

#     alpha_front = front_cropped[:, :, 3:4] / 255
#     alpha_back = back_cropped[:, :, 3:4] / 255

#     # replace an area in result with overlay
#     result = back.copy()
#     return result


# def make_smaller(img, scale_percent=15):
#     width = int(img.shape[1] * scale_percent / 100)
#     height = int(img.shape[0] * scale_percent / 100)

#     dsize = (width, height)

#     # resize image
#     small_image = cv2.resize(img, dsize)
#     return small_image


# def augment_selfie(img, backg_img):
#     img = make_smaller(img)
#     res = merge_image(backg_img, img, 150, 150)
#     return res


# def create_augmented_image(path, background):
#     folder, filename = os.path.split(path)
#     res = augment_selfie(cv2.imread(path), background)
#     os.chdir(folder)
#     name, extension = os.path.splitext(filename)
#     new_filename = f"{name}_aug{extension}"
#     cv2.imwrite(new_filename, res)


# def rewrite_to_augmented(path, background):
#     folder, filename = os.path.split(path)
#     res = augment_selfie(cv2.imread(path), background)
#     os.chdir(folder)
#     cv2.imwrite(filename, res)


# def extract_face(filename, required_size=(160, 160)):
#         """
#         Extract a single face from a given photograph
#         """
#         image = Image.open(filename)
#         # convert to RGB, if needed
#         image = image.convert('RGB')
#         # convert to array
#         pixels = asarray(image)
#         # create the detector, using default weights
#         detector = MTCNN()
#         # detect faces in the image
#         results = detector.detect_faces(pixels)
#         if len(results) == 0:
#             return
#         x1, y1, width, height = results[0]['box']
#         x1, y1 = abs(x1), abs(y1)
#         x2, y2 = x1 + width, y1 + height
#         # extract the face
#         face = pixels[y1:y2, x1:x2]
#         # resize pixels to the model size
#         image = Image.fromarray(face)
#         image = image.resize(required_size)
#         return asarray(image)

# folder = 'C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/check/dataset/train/pratham'
# i = 1
# # # # enumerate files
# print(listdir(folder))
# for filename in listdir(folder):
#     path = os.path.join(folder, filename)
#     face = extract_face(path)
#     print(path)
#     if face is None:
#       print(f'I can`t find a person in {filename}!\nI will try to use augmentation.\n')
#       back = cv2.imread('backg.jpg')
#       rewrite_to_augmented(path, back)
#       continue
#     print(i, face.shape)
#     # plot
#     pyplot.subplot(4, 7, i)
#     pyplot.axis('off')
#     pyplot.imshow(face)
#     i += 1
# pyplot.show()

# calculate a face embedding for each face in the dataset using facenet
# def get_embedding(model, face_pixels):
#         """Get the face embedding for one face"""
#         # scale pixel values
#         print(face_pixels.shape)
#         face_pixels = face_pixels.astype('float32')
#         # standardize pixel values across channels (global)
#         mean, std = face_pixels.mean(), face_pixels.std()
#         face_pixels = (face_pixels - mean) / std
#         # transform face into one sample
#         samples = expand_dims(face_pixels, axis=0)
        
#         # make prediction to get embedding
#         yhat = model.predict(samples)
#         # print(yhat)
        
#         return yhat[0]

# def face_to_embedings(faces, model):
#         """Convert each face in the train set to an embedding."""
#         embedings = []
#         for face_pixels in faces:
#             embedding = get_embedding(model, face_pixels)
#             embedings.append(embedding)
#         embedings = asarray(embedings)
#         return embedings


# train = load('face_train_dataset.npz')
# test = load('face_test_dataset.npz')

# train_X, train_Y, test_X, test_Y = train['arr_0'], train['arr_1'], test['arr_0'], test['arr_1']

# model = load_model(str(os.getcwd()) + '/facenet_keras.h5', compile=False)

# print(train_X.shape)

# newTrainX = face_to_embedings(train_X, model)
# newTestX = face_to_embedings(test_X, model)
# # savez_compressed(str(os.getcwd()) + '/face_train_embeddings.npz', newTrainX, train_Y)
# # savez_compressed(str(os.getcwd()) + '/face_test_embeddings.npz',  newTestX, test_Y)


# print('Loaded: ', train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
# print('Loaded Model')
# print(newTrainX.shape)
# print(newTestX.shape)


# train = load(str(os.getcwd()) + '/face_train_embeddings.npz')
# test = load(str(os.getcwd()) + '/face_test_embeddings.npz')
# train_X, train_Y, test_X, test_Y = train['arr_0'], train['arr_1'], test['arr_0'], test['arr_1']
# print(f'Dataset: train={train_X.shape[0]}, test={test_X.shape[0]}')

# # normalize input vectors
# in_encoder = Normalizer(norm='l2')

# # print(train_X)
# # print(test_X)
# train_X = in_encoder.transform(train_X)
# test_X = in_encoder.transform(test_X)

# # label encode targets
# out_encoder = LabelEncoder()
# out_encoder.fit(train_Y)
# train_Y = out_encoder.transform(train_Y)
# test_Y = out_encoder.transform(test_Y)
# model = SVC(kernel='linear', probability=True)
# model.fit(train_X, train_Y)
# yhat_train = model.predict(train_X)
# yhat_test = model.predict(test_X)
# score_train = accuracy_score(train_Y, yhat_train)
# score_test = accuracy_score(test_Y, yhat_test)

# print(f'Accuracy: train={score_train * 100:.3f}, test={score_test * 100:.3f}')

# test = load(str(os.getcwd()) + '/face_test_dataset.npz')
# testX_faces = test['arr_0']
# train = load(str(os.getcwd()) + '/face_train_embeddings.npz')
# test = load(str(os.getcwd()) + '/face_test_embeddings.npz')
# train_X, train_Y, test_X, test_Y = train['arr_0'], train['arr_1'], test['arr_0'], test['arr_1']
# in_encoder = Normalizer(norm='l2')
# train_X = in_encoder.transform(train_X)
# test_X = in_encoder.transform(test_X)
# out_encoder = LabelEncoder()
# out_encoder.fit(train_Y)
# train_Y = out_encoder.transform(train_Y)
# test_Y = out_encoder.transform(test_Y)
# model = SVC(kernel='linear', probability=True)
# model.fit(train_X, train_Y)



# selection = choice([i for i in range(test_X.shape[0])])
# random_face_pixels = testX_faces[selection]
# random_face_emb = test_X[selection]
# random_face_class = test_Y[selection]
# random_face_name = out_encoder.inverse_transform([random_face_class])
# samples = expand_dims(random_face_emb, axis=0)
# yhat_class = model.predict(samples)
# yhat_prob = model.predict_proba(samples)
# class_index = yhat_class[0]
# class_probability = yhat_prob[0,class_index] * 100
# predict_names = out_encoder.inverse_transform(yhat_class)
# print(f'Predicted: {predict_names[0]} {class_probability:.3f}')
# print(f'Expected: {random_face_name[0]}' )
# pyplot.imshow(random_face_pixels)
# title = f'{predict_names[0]} {class_probability:.3f}'
# pyplot.title(title)
# pyplot.show()


from numpy import asarray, savez_compressed, expand_dims, load
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot 
from sklearn.metrics import accuracy_score
from PIL import Image
from mtcnn.mtcnn import MTCNN

from matplotlib import pyplot 

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# from keras.models import load_model
# model = load_model(str(os.getcwd()) + '/facenet_keras.h5', compile=False)

# import cv2
# import numpy as np
# from PIL import Image 
# import PIL 
  

# def get_embedding(model, face_pixels):
#         """Get the face embedding for one face"""
#         # scale pixel values
#         face_pixels = face_pixels.astype('float32')
#         # standardize pixel values across channels (global)
#         mean, std = face_pixels.mean(), face_pixels.std()
#         face_pixels = (face_pixels - mean) / std
#         # transform face into one sample
#         samples = expand_dims(face_pixels, axis=0)
#         # make prediction to get embedding
#         yhat = model.predict(samples)
#         return yhat[0]
        
# def extract_face(filename, required_size=(160, 160)):
#         """
#         Extract a single face from a given photograph
#         """
#         image = Image.open(filename)
#         # convert to RGB, if needed
#         image = image.convert('RGB')
#         # convert to array
#         pixels = asarray(image)
#         # create the detector, using default weights
#         detector = MTCNN()
#         # detect faces in the image
#         results = detector.detect_faces(pixels)
#         if len(results) == 0:
#             return
          
#         faces = []

#         for i in range(len(results)) : 
#           x1, y1, width, height = results[i]['box']
#           x1, y1 = abs(x1), abs(y1)
#           x2, y2 = x1 + width, y1 + height
#           # extract the face
#           face = pixels[y1:y2, x1:x2]
#           # resize pixels to the model size
#           image = Image.fromarray(face)
#           image = image.resize(required_size)
#           # cv2_imshow(asarray(image))

#           faces.append(asarray(image))

#         return asarray(faces)

# face =  mpimg.imread(str(os.getcwd()) + '/pratham.jpg')
# imgplot = plt.imshow(face)
# plt.show()








# train = load(str(os.getcwd()) + '/face_train_embeddings.npz')
# test = load(str(os.getcwd()) + '/face_test_embeddings.npz')
# train_X, train_Y, test_X, test_Y = train['arr_0'], train['arr_1'], test['arr_0'], test['arr_1']
# in_encoder = Normalizer(norm='l2')
# train_X = in_encoder.transform(train_X)
# test_X = in_encoder.transform(test_X)
# out_encoder = LabelEncoder()
# out_encoder.fit(train_Y)
# train_Y = out_encoder.transform(train_Y)
# test_Y = out_encoder.transform(test_Y)







# face_model = load_model(str(os.getcwd()) + '/facenet_keras.h5', compile=False)
# face_pixels = extract_face(str(os.getcwd()) + '/multi.jpg')

# criminals = []

# for i in range(len(face_pixels)) : 

#   embedding = get_embedding(face_model, face_pixels[i])

#   # plt.imshow(face_pixels[i])
#   # plt.show()
#   # print(embedding)
#   # random_face_emb = in_encoder.transform(get_embedding(model,random_face_pixels))

#   # plt.imshow(face_pixels)
#   # plt.show()

#   # random_face_emb = get_embedding(model,random_face_pixels)
#   # print(random_face_pixels.shape)
#   # print(random_face_pixels.shape)



#   model = SVC(kernel='linear', probability=True)
#   model.fit(train_X, train_Y)

#   # random_face_class = test_Y[selection]
#   # random_face_name = out_encoder.inverse_transform([random_face_class])
#   # print(embedding)
#   samples = expand_dims(embedding, axis=0)
#   # print(samples)
#   yhat_class = model.predict(samples)
#   yhat_prob = model.predict_proba(samples)
#   class_index = yhat_class[0]
#   class_probability = yhat_prob[0,class_index] * 100
#   predict_names = out_encoder.inverse_transform(yhat_class)

#   gray = cv2.cvtColor(face_pixels[i], cv2.COLOR_BGR2GRAY)  
#   maxi = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))
#   # print("maxi = ",maxi)

#   if class_probability > 95 and maxi > 30 :
#     criminals.append(predict_names[0])
#     print(f'Predicted: {predict_names[0]} {class_probability:.3f}')

#     im1 = Image.fromarray(face_pixels[i]) 
#     im1 = im1.save(f"C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/check/{predict_names[0]}.jpg")

    

#     pyplot.imshow(face_pixels[i])
#     title = f'{predict_names[0]} {class_probability:.3f}'
#   else : 
#     print(f'Predicted: Unknown')
#     pyplot.imshow(face_pixels[i])
#     title = f'Unknown'


#   pyplot.title(title)
#   pyplot.show()


# import firebase_admin 
# from firebase_admin import firestore
# from firebase_admin import credentials

# cred = credentials.Certificate("criminal-detection-n-tracking-firebase-adminsdk-q3hv6-0a9f4f38d2.json")
# app = firebase_admin.initialize_app(cred,{'storageBucket': 'criminal-detection-n-tracking.appspot.com'})
# db = firestore.client()


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import requests

def sendmail(criminals,cam) : 
  msg = MIMEMultipart()
  fromaddr = "prathamtestmail10@gmail.com"
  msg['Subject'] = "Group of Criminals Detected"
  msg['From'] = fromaddr


  s = smtplib.SMTP('smtp.gmail.com', 587)
  # start TLS for security
  s.starttls()
  # Authentication
  s.login(fromaddr, "ondgozytpbtlfscu")

  mail = 'Criminals Detected : '

  for criminal in criminals : 
    # data = list(db.collection('Criminals').where('name','==',criminal).stream())
    data = requests.get('http://127.0.0.1:8000/api/crim/'+str(criminal))
    description = "#"
    imageURL = "#"
    print(data)

    try : 
      print(data.json())
      description = data.json()
      # if 'description' in data[0].to_dict() : 
      #   description = str(data[0].to_dict()['description'])
      if data.json()['refer'] : 
        imageURL = 'http://127.0.0.1:8000' + data.json()['refer']
        print(imageURL)
        r = requests.get(imageURL, stream=True)
        if r.status_code == 200:
            with open(str(os.getcwd()) + f'/{criminal}1.jpg', 'wb') as f:
                for chunk in r:
                    f.write(chunk)
        with open(str(os.getcwd()) + f'/{criminal}1.jpg', 'rb') as f:
          img_data = f.read()
          image = MIMEImage(img_data, name=str(os.getcwd()) + f'/{criminal}1.jpg')
          msg.attach(image)

          if data.json()['lat4'] != cam[3] and data.json()['longt4'] != cam[4] : 
            print("put request started")
            print(data.json())

            files = {'refer': open(f"C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/check/{criminal}1.jpg", 'rb')}

            response = requests.put('http://127.0.0.1:8000/api/crim/edit/'+str(criminal) + "/", data ={
                  'crims_id':data.json()['crims_id'],
                  'name' : data.json()['name'],
                  'height' : data.json()['height'],
                  'eyes' : data.json()['eyes'],
                  'skin' : data.json()['skin'],

                  'lat1' : data.json()['lat2'],
                  'longt1' : data.json()['longt2'],

                  'lat2' : data.json()['lat3'],
                  'longt2' : data.json()['longt3'],
                  
                  'lat3' : data.json()['lat4'],
                  'longt3' : data.json()['longt4'],

                  'lat4' : cam[3],
                  'longt4' : cam[4],


                },
                files = files)
            print(response)
            print("put request started")
        # image = requests.get(url=imageURL)
        # print("image = ",type(image))
        # cv2_imshow(image)
      
    except Exception as e : 
      description = "Not Available"
      imageURL = '#'

    mail = mail + f'''
    <div>
      <h6>Name : {criminal}</h6>
      <h6>Description : {description}</h6>
    </div>
    '''

    with open(str(os.getcwd()) + f'/{criminal}.jpg', 'rb') as f:
          img_data = f.read()
    image = MIMEImage(img_data, name=str(os.getcwd()) + f'/{criminal}.jpg')
    msg.attach(image)

  msg.attach(MIMEText(mail, 'html'))


  # emails = list(db.collection('referanizations').where('name','==','pratham').stream())
  # toaddr = emails[0].to_dict()['emails']

  toaddr = []

  try : 
    toaddr.append(cam[6])
  except Exception as e : 
    print(e)

  try : 
    toaddr.append(cam[7])
  except Exception as e : 
    print(e)

  if len(criminals) >= 3 : 
    try : 
      toaddr.append(cam[8])
    except Exception as e : 
      print(e)

  msg['To'] = ", ".join(toaddr)
  text = msg.as_string()
  s.sendmail(fromaddr, toaddr, text)


  s.quit()
      
def get_embedding(model, face_pixels):
        """Get the face embedding for one face"""
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]
        
def extract_face(image, required_size=(160, 160)):
        """
        Extract a single face from a given photograph
        """

        # print("shape = ",image.size)
        
        # image = Image.open(filename)
        # convert to RGB, if needed
        # image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        if len(results) == 0:
            return
          
        faces = []
        shapes = []

        for i in range(len(results)) : 
          x1, y1, width, height = results[i]['box']
          x1, y1 = abs(x1), abs(y1)
          x2, y2 = x1 + width, y1 + height
          # extract the face

          




          face = pixels[y1:y2, x1:x2]
          # resize pixels to the model size
          image = Image.fromarray(face)
          image = image.resize(required_size)
          # cv2.imshow("image1",asarray(image))

          # print("x1 = ",x1)
          # print("y1 = ",y1)
          # print("x2 = ",x2)
          # print("y2 = ",y2)

          # x1 = max(0,x1-50)
          # y1 = max(0,y1-50) 
          # x2 = min(x2+50,pixels.shape[1])
          # y2 = min(y2+50,pixels.shape[0])

          # print("shape[1] = ",pixels.shape[1])
          # print("shape[0] = ",pixels.shape[0])

          # print("x1 = ",x1)
          # print("y1 = ",y1)
          # print("x2 = ",x2)
          # print("y2 = ",y2)

          faces.append(asarray(image))
          shapes.append(asarray([x1,y1,x2,y2]))

        return [asarray(faces),asarray(shapes)]
from datetime import datetime


train = load(str(os.getcwd()) + '/face_train_embeddings.npz')
test = load(str(os.getcwd()) + '/face_test_embeddings.npz')
train_X, train_Y, test_X, test_Y = train['arr_0'], train['arr_1'], test['arr_0'], test['arr_1']
in_encoder = Normalizer(norm='l2')
train_X = in_encoder.transform(train_X)
test_X = in_encoder.transform(test_X)
out_encoder = LabelEncoder()
out_encoder.fit(train_Y)
train_Y = out_encoder.transform(train_Y)
test_Y = out_encoder.transform(test_Y)


model = SVC(kernel='linear', probability=True)
model.fit(train_X, train_Y)


criminals = []

face_model = load_model(str(os.getcwd()) + '/facenet_keras.h5', compile=False)

def recognize(pil_im,face_pixels,shapes) :
 
  

  for i in range(len(face_pixels)) : 
    # print("images 1 : ")
    x1,y1,x2,y2 = shapes[i]
    img = face_pixels[i][y1:y2,x1:x2]
    # cv2.imshow("image2",face_pixels[i])

    embedding = get_embedding(face_model, face_pixels[i])

    # plt.imshow(face_pixels[i])
    # plt.show()
    # print(embedding)
    # random_face_emb = in_encoder.transform(get_embedding(model,random_face_pixels))

    # plt.imshow(face_pixels)
    # plt.show()

    # random_face_emb = get_embedding(model,random_face_pixels)
    # print(random_face_pixels.shape)
    # print(random_face_pixels.shape)





    # random_face_class = test_Y[selection]
    # random_face_name = out_encoder.inverse_transform([random_face_class])
    # print(embedding)
    samples = expand_dims(embedding, axis=0)
    # print(samples)
    face_pixels[i] = cv2.cvtColor(face_pixels[i],cv2.COLOR_BGR2RGB)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)

    gray = cv2.cvtColor(face_pixels[i], cv2.COLOR_BGR2GRAY)  
    maxi = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))
    # print("maxi = ",maxi)
    

    
    

    if maxi/class_probability > 10 :
      
      print(f'Predicted: {predict_names[0]} {class_probability:.3f}')

      if predict_names[0] not in criminals : 

        pyplot.imshow(face_pixels[i])
        title = f'{predict_names[0]} {class_probability:.3f} {maxi}'
        pyplot.title(title)
        pyplot.show()

        criminals.append(predict_names[0])




      pixels = asarray(pil_im)
      im1 = Image.fromarray(pixels[y1:y2,x1:x2]) 
      # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
      im1 = im1.save(str(os.getcwd()) + f"/{predict_names[0]}.jpg")

      
      
      # pyplot.imshow(face_pixels[i])
      title = f'{predict_names[0]} {class_probability:.3f} {maxi}'
      print(title)
      # print("images 2 : ")
      # pyplot.title(title)
      # pyplot.show()
    # else : 
    #   print(f'Predicted: Unknown {predict_names[0]} {class_probability:.3f} {maxi}')
    #   pyplot.imshow(face_pixels[i])
    #   title = f'Unknown'
    #   pyplot.title(title)
    #   pyplot.show()

    
  
  


import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("https://10.40.12.251:8080/video")
# cap = cv2.VideoCapture('rtsp://student:student123@192.168.3.12')

def runCam(cam) :

  # cap = cv2.VideoCapture(str(cam[2]))


  # fps = int(cap.get(cv2.CAP_PROP_FPS))
  save_interval = 5
  frame_count = 0
  # fps = int(cap.get(cv2.CAP_PROP_FPS))
  
  # Check if camera opened successfully
  if (cap.isOpened()== False):
      print("Error opening video file")
 
  # # Read until video is completed
  while(cap.isOpened()):
      
  # # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True :

        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

        frame_count += 1

        # if frame_count % (fps * save_interval) == 0 :
        if (datetime.now().second%2 == 0) :
          cv2.imwrite(str(os.getcwd()) + f'/images/{frame_count}.jpg',frame)
          pil_im = Image.fromarray(frame)
          fs = extract_face(pil_im)

          if fs is not None : 
            recognize(frame,fs[0],fs[1])
          # optional 
          frame_count = 0
        
        # print("criminals ",criminals)

        if len(criminals) > 0 and (datetime.now().second%15 == 0): 
          print("Time = ",datetime.now().second)
          sendmail(criminals,cam)
          print("Mail sent !!!!!! ")
          for criminal in criminals : 
            data = requests.get('http://127.0.0.1:8000/api/crim/'+str(criminal))
            print(data)
            # if data.response_code != 404 :
            #   data = data.json()
            #   

      else : 
        break 
      

  #     # Display the resulting frame

          # frame = cv2.resize(frame,(160,160))
          # converted = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
          # 
          # 

          # 
          # cv2_imshow( frame)
          # face =  mpimg.imread(str(os.getcwd()) + '/pratham.jpg')
          # imgplot = plt.imshow(face)
          # plt.show()
          
  #     # Press Q on keyboard to exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
  
  # # Break the loop
    
  
  # # When everything done, release
  # # the video capture object

  # sendmail(criminals)

  # print("Mail sent !!!!!! ")
  # for criminal in criminals : 
  #   data = requests.get('http://127.0.0.1:8000/api/crim/'+str(criminal)).json()
  #   response = requests.put('http://127.0.0.1:8000/api/crim/edit/'+str(criminal), data ={
  #       'crims_id':criminal,
  #       'name' : data['name'],
  #       'height' : data['height'],
  #       'eyes' : data['eyes'],
  #       'skin' : data['skin'],
  #       'lat1' : data['lat2'],
  #       'longt1' : data['longt2'],
  #       'lat2' : data['lat3'],
  #       'longt2' : data['longt3'],
        
  #       'lat3' : data['lat4'],
  #       'longt3' : data['longt4'],
  #       'lat4' : '12345',
  #       'longt4' : '12345',
  #       'refer' : Image.open(f"C:/Users/Prath/OneDrive/Desktop/Check files/Mini Project 1/check/{criminal}1.jpg")
  #     })

  cap.release()
  
  # # Closes all the frames
  cv2.destroyAllWindows()