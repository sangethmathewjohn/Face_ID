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

#### Training

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




