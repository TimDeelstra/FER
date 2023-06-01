import cv2
import numpy as np
from time import sleep
from keras.utils import img_to_array
from keras.models import load_model
from retinaface import RetinaFace
from deepface import DeepFace

# https://www.kaggle.com/code/yasserhessein/emotion-recognition-with-resnet50/notebook

# Decent performance, showing decent fear/sad/surprise correlation. Might be useful as positive/negative emotion classifier.
# About 12ms for ryzen laptop.


classifier = load_model("./models/hse/mobilenet_7.h5")
target_size = [224,224]
model_name = "VGG-Face"
render = True
frame_threshold = 1

class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# class_labels = {v: k for k, v in class_labels.items()}
# classes = list(class_labels.values())


offset = 4

def face_detector(face, img):
    # Convert image to grayscale
    box = face["facial_area"]
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
    roi = img[y_min:y_max, x_min:x_max]

    try:
        roi_gray = cv2.resize(roi, (224, 224), interpolation = cv2.INTER_AREA)
    except:
        return (x_min,bbox_width,y_min,bbox_height), np.zeros((48,48), np.uint8), img
    return (x_min,bbox_width,y_min,bbox_height), roi_gray, img

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("../../Downloads/Proefpersoon51014_Sessie1.MP4")

while True:

    ret, frame = cap.read()
    faces = RetinaFace.detect_faces(frame)
    if faces is not None:
        print(faces)
        for face in faces:
            rect, face, image = face_detector(faces[face], frame)
            if np.sum([face]) != 0.0:
                roi = face.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class
                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]  
                label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
                cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)

                image[0:face.shape[0], 0:face.shape[1]] = face
        #else:
            #cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
        
    cv2.imshow('All', image)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows() 