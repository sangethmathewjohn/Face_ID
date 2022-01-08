
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
