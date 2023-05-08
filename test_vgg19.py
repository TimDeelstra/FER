import cv2
import numpy as np
from time import sleep
from keras.utils import img_to_array
from keras.models import load_model
from face_detection import RetinaFace
from keras.models import model_from_json


# https://www.kaggle.com/code/enesztrk/facial-emotion-recognition-vgg19-fer2013

# Decent performance, showing decent neutral/surprise correlation. Does not often confuse neutral with suprise/fear.
# Might be useful as positive/neutral/negative emotion classifier.
# About 18ms for ryzen laptop.


classifier = model_from_json(open("models/vgg19/model.yaml", "r").read())
classifier.load_weights('models/vgg19/model.h5')

class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# class_labels = {v: k for k, v in class_labels.items()}
# classes = list(class_labels.values())

detector = RetinaFace()

offset = 4

def face_detector(face, img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    box, _, _ = face
    x_min=int(box[0])-offset
    if x_min < 0:
        x_min = 0
    y_min=int(box[1])-offset
    if y_min < 0:
        y_min = 0
    x_max=int(box[2])+offset
    y_max=int(box[3])+offset
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255,0,0),2)
    roi_gray = gray[y_min:y_max, x_min:x_max]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
    except:
        return (x_min,bbox_width,y_min,bbox_height), np.zeros((48,48), np.uint8), img
    return (x_min,bbox_width,y_min,bbox_height), roi_gray, img

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    faces = detector(frame)
    if faces is not None:
        for face in faces:
            rect, face, image = face_detector(face, frame)
            if np.sum([face]) != 0.0:
                
                img = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
                img = img.astype("float") / 255.0
                img = np.expand_dims(img, axis=0)
                

                # make a prediction on the ROI, then lookup the class
                preds = classifier.predict(img)[0]
                label = class_labels[preds.argmax()]  
                label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
                cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
        #else:
            #cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
        
    cv2.imshow('All', image)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows() 