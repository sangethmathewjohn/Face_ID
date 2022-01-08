import os
import numpy as np 
from PIL import Image
import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

current_id=0
label_ids={}
y_labels = []
x_train=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.join(root,file).replace(" ","-").lower()
            print(label,path)
            if label in label_ids:
                pass
            else:
                label_ids[label] =current_id
                current_id += 1
            id_ =label_ids[label]
            print(label_ids)

            #y_labels.append(label)
            #x_labels.append(path)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image,"uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                



