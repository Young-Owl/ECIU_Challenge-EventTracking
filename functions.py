import cv2
import csv
import threading

# Start a new CamThread, inside it, invoke the preview of the camera.
class camThread(threading.Thread):
    def __init__(self, previewName, camID, Cascade, Scale, MinNeighbors):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.Cascade = Cascade
        self.Scale = Scale
        self.MinNeighbors = MinNeighbors
    def run(self):
        print ("Opening " + self.previewName)
        camPreview(self.previewName, self.camID, self.Cascade, self.Scale, self.MinNeighbors)

def camPreview(previewName, camID, Cascade, Scale, MinNeighbors):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    fps = cam.get(cv2.CAP_PROP_FPS)
    ppl_count = 0
    diff_ppl = True
    time = 0

    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        rval, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        i = 0
        
        for param in Cascade[0]:
            person_detected = param.detectMultiScale(gray, Scale, MinNeighbors)

            try:
                human_count = person_detected.shape[0]
            except:
                human_count = 0
                time += 1
            else:
                time = 0
                if(diff_ppl):
                    ppl_count += human_count
                    get_coords(0, person_detected)
                    diff_ppl = False
            break

        if(human_count == 0 and time == 100):
            diff_ppl = True
        
        i += 1
        if i>= len(Cascade[1]):
            i = 0
        
        draw(frame, person_detected, Cascade[1][i])
        cv2.putText(frame, "FPS: " + str(fps), (0, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), bottomLeftOrigin=False)
        cv2.putText(frame, "PPL: " + str(ppl_count), (250, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), bottomLeftOrigin=False)
        #check_pos(frame, person_detected)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
        
        cv2.imshow(previewName, frame)
    cam.release()
    cv2.destroyWindow(previewName)

def getClassifier():
    classifier = []

    # SELECT all the classifiers used, from big to small
    #fullbody = cv2.CascadeClassifier('haarcascade/haarcascade_fullbody.xml')
    #upperbody = cv2.CascadeClassifier('haarcascade/haarcascade_upperbody.xml')
    frontalface_default = cv2.CascadeClassifier('C:/Users/young/OneDrive/Ambiente de Trabalho/head-detector-master/cascades/cascade_front.xml')
    back = cv2.CascadeClassifier('C:/Users/young/OneDrive/Documentos/VSCode/ECIU_Challenge/cascades/cascade_back.xml')

    #classifier.append(fullbody)
    #classifier.append(upperbody)
    classifier.append(frontalface_default)
    classifier.append(back)

    #classifier_name = ["fullbody", "upperbody", "frontalface_default", "back"]
    classifier_name = ["frontalface_default", "back"]

    # TODO: Change to dictionary
    return classifier, classifier_name

def get_coords(camId, person):
    for (x, y, w, h) in person:
        coords = {"x": int(x+(w/2)), "y": int(y+(h/2))}
        
        with open("data.csv", mode="a") as data:
            csv_writer = csv.writer(data, delimiter=",")
            csv_writer.writerow([camId, coords["x"], coords["y"]])

def draw(frame, person, name):
    for (x, y, w, h) in person:
        cv2.rectangle(frame,        # img
                      (x,y),        # pt1
                      (x+w, y+h),   # pt2
                      (255, 0, 0),  # color
                      2)            # thickness
        cv2.putText(frame,                      # img
                    name,                       # text
                    (x,y-5),                    # org
                    cv2.FONT_HERSHEY_COMPLEX,   # font name
                    0.7,                        # font scale
                    (0, 0, 255))                # color

def check_pos(frame, person):
    for (x, y, w, h) in person:
        cv2.line(frame,
                 (int(x+(w/2)), y),
                 (int(x+(w/2)), y+h),
                 (255, 105, 180),
                 2)
        cv2.line(frame,
                 (x, int(y+(h/2))),
                 (x+w, int(y+(h/2))),
                 (255, 105, 180),
                 2)