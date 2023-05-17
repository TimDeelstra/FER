import cv2
import numpy as np
import time
from keras.utils import img_to_array
from keras.models import load_model
from face_detection import RetinaFace
from keras.models import model_from_json


# https://www.kaggle.com/code/enesztrk/facial-emotion-recognition-vgg19-fer2013

# Decent performance, showing decent neutral/surprise correlation. Does not often confuse neutral with suprise/fear.
# Might be useful as positive/neutral/negative emotion classifier.
# About 18ms for ryzen laptop.
# About 9ms for katana.


classifier = model_from_json(open("models/vgg19/model.yaml", "r").read())
classifier.load_weights('models/vgg19/model.h5')

class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# class_labels = {v: k for k, v in class_labels.items()}
# classes = list(class_labels.values())

detector = RetinaFace(gpu_id=0)

offset = 4

def face_detector(face, img):
    # Convert image to grayscale
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
    #cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255,0,0),2)
    


    try:
        roi_gray = cv2.resize(img, (48, 48), interpolation = cv2.INTER_AREA)
    except:
        return (x_min,bbox_width,y_min,bbox_height), np.zeros((48,48), np.uint8), img
    return (x_min,bbox_width,y_min,bbox_height), roi_gray

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../../Downloads/Proefpersoon51014_Sessie1.MP4")

while True:
    detects = []
    start = time.time()
    while len(detects) < 32:
        ret, frame = cap.read()
        faces = detector(frame)
        if faces is not None:
            face = faces[0]
            if(face[0][2]-face[0][0] > 120):
                rect, face = face_detector(face, frame)
                if np.sum([face]) != 0.0:
                    
                    img = face
                    img = img.astype("float") / 255.0
                    img = np.expand_dims(img, axis=0)

                    detects.append([frame, img, rect])
                    
    print(time.time()-start)
                    # make a prediction on the ROI, then lookup the class
    start = time.time()
    preds = classifier.predict(np.vstack(np.array(detects)[0:,1]),batch_size=32)
    print(time.time()-start)
    # for i in np.arange(len(preds)):
    #     pred = preds[i]
    #     frame, img, rect = detects[i]
    #     label = class_labels[pred.argmax()]  
    #     label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
    #     cv2.rectangle(frame,(rect[0],rect[2]),(rect[0]+rect[1],rect[2]+rect[3]),(255,0,0),2)
    #     cv2.putText(frame, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    #         #else:
    #             #cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
        
    #     cv2.imshow('All', frame)
    #     if cv2.waitKey(1) == 13: #13 is the Enter Key
    #         break
        
cap.release()
cv2.destroyAllWindows() 