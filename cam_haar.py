import numpy as np
import cv2, os

#front_cascade = cv2.CascadeClassifier('C:/Users/young/OneDrive/Ambiente de Trabalho/head-detector-master/cascades/cascade_front.xml')
#side_cascade = cv2.CascadeClassifier('C:/Users/young/OneDrive/Ambiente de Trabalho/head-detector-master/cascades/cascade_side.xml')
#back_cascade = cv2.CascadeClassifier('C:/Users/young/OneDrive/Ambiente de Trabalho/head-detector-master/cascades/cascade_back.xml')

cascPedestrian = cv2.CascadeClassifier('C:/Users/young/OneDrive/Documentos/VSCode/ECIU_Challenge/cascades/cascade_back.xml')

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
while True:
    ret, img = cap.read()
    if ret == True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        face_front = cascPedestrian.detectMultiScale(gray, 1.2, 6)

        skip = False

        if len(face_front) > 0:
            for (x,y,w,h) in face_front:
                cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = gray[y:y+h, x:x+w]
            skip = True

        cv2.imshow('img',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
