import cv2
import numpy as np
from time import sleep
from keras.utils import img_to_array
from keras.models import load_model
from face_detection import RetinaFace
import keras.backend as K
import tensorflow as tf

# https://www.kaggle.com/code/yasserhessein/emotion-recognition-with-resnet50/notebook

# Decent performance, showing decent fear/sad/surprise correlation. Might be useful as positive/negative emotion classifier.
# About 12ms for ryzen laptop.

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc'),
      f1_score,
]

dependencies = {
    'metrics': METRICS,
    'f1_score': f1_score,
}

classifier = load_model("./models/EfficientNet/model.h5", custom_objects=dependencies)

class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# class_labels = {v: k for k, v in class_labels.items()}
# classes = list(class_labels.values())

detector = RetinaFace(gpu_id=0)

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
                roi = face.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class
                preds = classifier.predict(roi)[0]
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