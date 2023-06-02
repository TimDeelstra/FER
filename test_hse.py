import cv2
import numpy as np
from time import sleep, time
from keras.utils import img_to_array
from keras.models import load_model
from retinaface import RetinaFace
from deepface import DeepFace
from hsemotion.facial_emotions import HSEmotionRecognizer

# https://www.kaggle.com/code/yasserhessein/emotion-recognition-with-resnet50/notebook

# Decent performance, showing decent fear/sad/surprise correlation. Might be useful as positive/negative emotion classifier.
# About 12ms for ryzen laptop.



target_size = (224,224)
model_name = 'enet_b0_8_best_afew'
render = True

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]

class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# class_labels = {v: k for k, v in class_labels.items()}
# classes = list(class_labels.values())

fer=HSEmotionRecognizer(model_name=model_name,device='cuda')


offset = 4

def face_detector(face, img):
    # Convert frame to grayscale
    box = face["facial_area"]
    x_min=int(box['x'])-offset
    if x_min < 0:
        x_min = 0
    y_min=int(box['y'])-offset
    if y_min < 0:
        y_min = 0
    x_max=int(box['x']) + int(box['w'])+offset
    y_max=int(box['y']) + int(box['h'])+offset
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255,0,0),2)
    roi = img[y_min:y_max, x_min:x_max]

    try:
        roi_gray = cv2.resize(roi, target_size, interpolation = cv2.INTER_AREA)
    except:
        return (x_min,bbox_width,y_min,bbox_height), np.zeros((48,48), np.uint8), img
    return (x_min,bbox_width,y_min,bbox_height), roi_gray, img

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../../Downloads/Proefpersoon51014_Sessie1.MP4")

while True:

    ret, frame = cap.read()
    rf = time()
    faces = DeepFace.extract_faces(img_path=frame, 
                target_size = (224, 224), 
                detector_backend = backends[4],
                enforce_detection=False)
    print(faces)
    print("RF: ", time()-rf)
    pred = time()
    if faces is not None:
        for face in faces:
            if face["confidence"] > 0.9:
                rect, face, frame = face_detector(face, frame)
                if np.sum([face]) != 0.0:
                    roi = face.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # make a prediction on the ROI, then lookup the class
                    # preds = classifier.predict(roi)[0]
                    # print(preds)
                    # label = class_labels[preds.argmax()]  
                    label,scores=fer.predict_emotions(roi,logits=True)
                    print(scores)
                    label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
                    cv2.putText(frame, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)

                    frame[0:face.shape[0], 0:face.shape[1]] = face
            else:
                cv2.putText(frame, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    else:
        cv2.putText(frame, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    print("Pred: ", time()-pred)   
    cv2.imshow('All', frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows() 