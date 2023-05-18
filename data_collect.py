import cv2
import os
from numpy import asarray, savez_compressed, expand_dims, load
from random import choice
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot 
from sklearn.metrics import accuracy_score
from PIL import Image
from mtcnn.mtcnn import MTCNN

# from matplotlib import pyplot 

# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt

video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count=0

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

nameID=str(input("Enter your name")).lower()
path='images/'+nameID

isExist=os.path.exists(path)

if isExist:
    print("Name Already Taken")
    nameID=str(input("enter your name again"))
else:
    os.makedirs(path)

while True:
    ret,frame=video.read()

    image= Image.fromarray(frame)

    pil_im = Image.fromarray(frame)
    fs = extract_face(pil_im)
    if fs is not None :
        for x1,y1,x2,y2 in fs[1]:
            count=count+1
            name='./images/'+nameID+'/'+str(count)+'.jpg'
            cv2.imwrite(name,frame[y1:y2,x1:x2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.imshow("WindowFrame",frame)
    cv2.waitKey(1)
    if count> 100 :
        break
video.release()
cv2.destroyAllWindows()



