import os
import numpy as np
from PIL import Image
import cv2

path ='images\\sangeeth'
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

def getImagesAndLabels(path):
    imagesPaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples =[]
    ids=[]
    for imagePath in imagesPaths:
        PIL_img =Image.open(imagePath).convert("L")
        img_numpy=np.array(PIL_img,"uint8")

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces=face_cascade.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print("starting")
faces,ids=getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.save("reco/trainer.yml")
print("done")
