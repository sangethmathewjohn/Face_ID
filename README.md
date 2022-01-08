# Face_ID

## Pre-requisite

    pip install opencv-python
    
    pip install pillow
    
    pip install opencv-contrib-python
 
## Step 1

#### Live face recognition

Drawing a rectangle around the face.

      
      import cv2
      import numpy as np


      face_cascade = cv2.CascadeClassifier(
          cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
     

      cap = cv2.VideoCapture(0)

      while(True):
          # Capture frame-by-frame
          ret, frame = cap.read()
          gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
          faces=face_cascade.detectMultiScale(gray,1.1,5)
          for (x,y,w,h) in faces:
              print(x,y,w,h)
              roi_gray = gray[y:y+h, x:x+w] 
              roi_color= frame[y:y+h, x:x+w]
              img_item = "7.png"
              cv2.imwrite(img_item, roi_color)
              color = (255, 0, 0)  # BGR 0-255
              stroke = 2
              end_cord_x = x + w
              end_cord_y = y + h
              cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

          cv2.imshow('frame', frame)
          if cv2.waitKey(20) & 0xFF == ord('q'):
              break

      # When everything done, release the capture
      cap.release()
      cv2.destroyAllWindows()

## Step 2

#### Capturing the training datsets


        import cv2


        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


        cap = cv2.VideoCapture(0)
        id =input('enter user id\n')
        sample =0
        while(True):
            # Capture frame-by-frame
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,minNeighbors=5)
            for (x, y, w, h) in faces:
                sample +=1
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                img_item = "7.png"
                cv2.imwrite("images/sangeeth/User"+str(id)+'.'+str(sample)+".jpg", roi_gray)
                color = (255, 0, 0)  # BGR 0-255
                stroke = 3
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)

            cv2.imshow('Face', img)
            cv2.waitKey(1)
            if sample>150:
                break
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


## Step 3

#### Training

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





