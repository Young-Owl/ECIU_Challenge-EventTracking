#   Python Learning - Gonçalo Rodrigues 98322 EET
#       ECIU Challenge - Group Learning Objetives
#       CTE Module - António J.R. Neves (an@ua.pt)
#       
#       All Python Material:
#       https://padlet.com/an1185/awaya3ic26i6kq5k

import cv2
import sys

# Get user supplied values
cascPathF = "C:/Users/young/OneDrive/Documentos/VSCode/ECIU_Challenge/cascades/haarcascade_frontalface_default.xml"
cascPathP = "C:/Users/young/OneDrive/Documentos/VSCode/ECIU_Challenge/cascades/haarcascade_profileface.xml"
cascPathB = "C:/Users/young/OneDrive/Documentos/VSCode/ECIU_Challenge/cascades/cascade_back.xml"


# Create the haar cascade
facefrontCascade = cv2.CascadeClassifier(cascPathF)
faceprofileCascade = cv2.CascadeClassifier(cascPathP)
facebackCascade = cv2.CascadeClassifier(cascPathB)

# Open Webcam
video_capture = cv2.VideoCapture(0)
video_capture1 = cv2.VideoCapture(2)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
fps = video_capture.get(cv2.CAP_PROP_FPS)


while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture.read()
    ret1, frame1 = video_capture1.read()
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    faceF = facefrontCascade.detectMultiScale(
        gray0,
        scaleFactor = 1.4,
        minNeighbors = 6,
        minSize = (50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    faceP = faceprofileCascade.detectMultiScale(
        gray0,
        scaleFactor = 1.4,
        minNeighbors = 6,
        minSize = (50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    faceB = facebackCascade.detectMultiScale(
        gray0,
        scaleFactor = 1.4,
        minNeighbors = 6,
        minSize = (50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangle:
    for (x,y,w,h) in faceF:
        cv2.rectangle(frame0,(x,y), (x+w, y+h), (36,255,12), 2)
        cv2.putText(frame0,'Front',(x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (36,255,12),2)

    for (x,y,w,h) in faceP:
        cv2.rectangle(frame0,(x,y), (x+w, y+h), (36,255,12), 2)
        cv2.putText(frame0,'Profile',(x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (36,255,12),2)
    
    for (x,y,w,h) in faceB:
        cv2.rectangle(frame0,(x,y), (x+w, y+h), (36,255,12), 2)
        cv2.putText(frame0,'Back',(x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (36,255,12),2)
    


    faceF = facefrontCascade.detectMultiScale(
        gray1,
        scaleFactor = 1.3,
        minNeighbors = 6,
        minSize = (50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    faceP = faceprofileCascade.detectMultiScale(
        gray1,
        scaleFactor = 1.3,
        minNeighbors = 6,
        minSize = (50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    faceB = facebackCascade.detectMultiScale(
        gray1,
        scaleFactor = 1.3,
        minNeighbors = 6,
        minSize = (50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangle:
    for (x,y,w,h) in faceF:
        cv2.rectangle(frame1,(x,y), (x+w, y+h), (36,255,12), 2)
        cv2.putText(frame1,'Front',(x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (36,255,12),2)

    for (x,y,w,h) in faceP:
        cv2.rectangle(frame1,(x,y), (x+w, y+h), (36,255,12), 2)
        cv2.putText(frame1,'Profile',(x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (36,255,12),2)
    
    for (x,y,w,h) in faceB:
        cv2.rectangle(frame1,(x,y), (x+w, y+h), (36,255,12), 2)
        cv2.putText(frame1,'Back',(x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (36,255,12),2)



    cv2.imshow('Video0', frame0)
    cv2.imshow('Video1', frame1)
    if cv2.waitKey(1) & 0xFF == 0X1B:
        break

video_capture.release()
video_capture1.release()
cv2.destroyAllWindows()