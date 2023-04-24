from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import os
import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import cv2
from face_detection import RetinaFace
import face_recognition
from keras.models import load_model
import matplotlib.colors as mcolors

def pprint(model, outputs, st, et):

    print(outputs)
    print("\nProcessing time:" + str(et-st) + "\n")

    logits = outputs.logits.split(1)
    for logit in logits:
        index = 0
        for value in logit[0]:
            print(model.config.id2label[index] + " = " + str(value.item()))
            index = index + 1
        print(model.config.id2label[logit.argmax(-1).item()] + "\n")

# basic processing (only resizing)
def process(examples):
    examples.update(extractor(examples['image'], ))
    return examples

def run(examples):
   examples.update(model(examples['pixel_values'], ))
   return examples

# p_dataset = dataset.map(process, batched=True)

# with torch.no_grad():
#     outputs = p_dataset.map(run, batched=True)

extractor = AutoFeatureExtractor.from_pretrained("kdhht2334/autotrain-diffusion-emotion-facial-expression-recognition-40429105176")

model = AutoModelForImageClassification.from_pretrained("kdhht2334/autotrain-diffusion-emotion-facial-expression-recognition-40429105176")

# model = load_model("./models/model_v6_23.hdf5")


cam = 0
cap = cv2.VideoCapture(cam)

#detector = RetinaFace(gpu_id=0)
detector = RetinaFace()

# images = ["2023-04-14-171515.jpg", "2023-04-15-192706.jpg", "2023-04-15-192720.jpg"]
# dataset = datasets.Dataset.from_dict({"image": images}).cast_column("image", datasets.Image())

# image = dataset["image"]

# inputs = extractor(image, return_tensors="pt")
# print(inputs["pixel_values"][0][0].shape)

colors = [mcolors.BASE_COLORS[name] for name in list(mcolors.BASE_COLORS)]




with torch.no_grad():
    while True:
            st = time.time()

            success, frame = cap.read()
            faces = detector(frame)

            offset = 4
            if faces is not None:
                index=0
                for box, landmarks, score in faces:
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

                    img = frame[y_min:y_max, x_min:x_max]
                    
                    inputs = extractor(img, return_tensors="pt")
                    outputs = model(**inputs)
                    # predicted_class = np.argmax(model.predict(img))
                    
                    et = time.time()
                    #pprint(model,outputs, st, et)
                    logits = outputs.logits.split(1)
                    for logit in logits:
                        color = tuple(255*x for x in colors[index])
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 1)
                        cv2.putText(frame, model.config.id2label[logit.argmax(-1).item()], (100,(index+1)*50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        index = index + 1

        

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            success,frame = cap.read()  
    



# pprint(model,outputs, st, et)

