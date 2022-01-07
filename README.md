# Face_ID

## Pre-requisite

    pip install opencv-python
 
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

