import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
import csv
import sys, getopt
from hsemotion.facial_emotions import HSEmotionRecognizer
from batch_face import RetinaFace
from rmn import RMN
import torch
from POSTER_V2.models.PosterV2_7cls import *
from POSTER_V2.models.train_func import *
from APViT.RAF import *
import mmcv
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcls.models import build_classifier
from mmcls.core import wrap_fp16_model
from mmcls.datasets.raf import FER_CLASSES
from mmcls.datasets.pipelines import Compose
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
import datetime
import torchvision.transforms as transforms
from PIL import Image


# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

offset = 0
target_size = (224, 224)
if torch.cuda.is_available():
    detector = RetinaFace(gpu_id=0)
else:
    detector = RetinaFace()


# pylint: disable=too-many-nested-blocks

def face_detector(box, img):
    # Convert frame to grayscale
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
    roi = img[y_min:y_max, x_min:x_max]

    try:
        roi_gray = cv2.resize(roi, target_size, interpolation = cv2.INTER_AREA)
    except:
        return (x_min,bbox_width,y_min,bbox_height), np.zeros((48,48), np.uint8), img
    return (x_min,bbox_width,y_min,bbox_height), roi_gray, img



def analysis(
    database,
    dir,
    file,
    model_name="VGG-Face",
    detector_backend="retinaface",
    enable_face_analysis=True,
    frame_threshold=1,
):
    # global variables
    pivot_img_size = 112  # face recognition result image

    enable_emotion = True
    # ------------------------
    # find custom values for this input set
    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    if(model_name == "VGG-Face"):
        DeepFace.build_model(model_name=model_name)
        print(f"facial recognition model {model_name} is just built")

        if enable_face_analysis:
            # DeepFace.build_model(model_name="Age")
            # print("Age model is just built")
            # DeepFace.build_model(model_name="Gender")
            # print("Gender model is just built")
            DeepFace.build_model(model_name="Emotion")
            print("Emotion model is just built")
    # visualization
    start = time.time()

    source = dir + "/" + file

    if(dir==""):
        source = file

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    v_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    resolution_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    resolution_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    num_lines = 0
    
    print("Resolution: ", resolution_x, "x", resolution_y)
    print("Framecount: ", str(v_length))
    print("FPS:" + str(fps) + "\n\n")
    

    framewaittime = 1
    if(rtplayback):
        framewaittime = 1000/fps


    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        exit(0)
    else:
        # Create data directory
        path = os.getcwd() + "/data"
        filename = file.split("/")[-1] + "." + model_name + "." + detector_backend + ".csv"
        for d in file.split("/")[:-1]:
            try:
                path += "/" + d
                os.mkdir(path)
            except OSError as error:
                if verbose:
                    print(error)  
        
        # Open data file in read/write
        try:
            f = open(path + "/" + filename, "a+")
            f.seek(0)
            reader = csv.reader(f)
            writer = csv.writer(f)
            num_lines = sum(1 for _ in f)
            f.seek(0)
        except IOError as e:
            print ("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)
        except: #handle other exceptions such as attribute errors
            print ("Unexpected error:", sys.exc_info()[0])
            exit(1)

    if(num_lines < v_length or render):
        while(cap.isOpened()):
            if verbose:
                print("fps: " + str(batch_size/(time.time()-start)))
            start = time.time()

            #LINLIN: CREATING BATCH
            batch_len = 0
            frames = []
            while batch_len < batch_size:
                _, img = cap.read()

                if img is None:
                    break

                frames.append(img)
                batch_len = batch_len + 1
            
            if frames == []:
                if verbose:
                    print("No more frames, end of file.")
                    print("Results stored in ", filename)
                break
            if verbose:
                print(str(100*cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT)) + "%% completed      ", end="\n")
            else:
                print(str(100*cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT)) + "%% completed      ", end="\r")
            #audio_frame, val = player.get_frame()

            fromStorage = -1

            demographies = []

            # Try loading the model results from storage
            try:
                for img in frames:
                    data = next(reader)
                    if verbose:
                        print("frame found:" + str(reader.line_num))
                    x, y, w, h, angry, disgust, fear, happy, sad, surprise, neutral, dominant = data
                    demography = {'emotion': {'angry': float(angry), 'disgust': float(disgust), 'fear': float(fear), 'happy': float(happy), 'sad': float(sad), 'surprise': float(surprise), 'neutral': float(neutral)}, 'dominant_emotion': dominant, 'region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}}
                    demographies.append(demography)
                    fromStorage = fromStorage + 1
                    if verbose:
                        print("frame loaded successfully")   
            # Run the model for the frames for which we didn't find results from storage   
            except StopIteration:
                start = time.time()
                # just extract the regions to highlight in webcam
                if(model_name == "VGG-Face"):
                    for i in range(fromStorage+1, batch_size):
                        try:
                            img = frames[i]
                            demography = DeepFace.analyze(
                                            img_path=img,
                                            detector_backend=detector_backend,
                                            enforce_detection=False,
                                            silent=True,
                                            actions="emotion",
                                            align=True
                                        )
                            #print(demography)
                            demographies.append(demography)
                        except:  # to avoid exception if no face detected
                            if verbose:
                                print("No face detected")
                            demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                # print(time.time()-start)
                if(model_name == "enet_b0_8_best_afew"):
                    faces = detector(frames, cv=False) #LINLIN: BATCH FRAMES
                    #print(time.time()-start)
                    for i in range(fromStorage+1, batch_size):
                        try:
                            box, landmarks, score = faces[i][0]
                            if score > 0.95:
                                rect, face, img = face_detector(box, frames[i])
                                if np.sum([face]) != 0.0:
                                    label,scores=fer.predict_emotions(face,logits=True)
                                    #print(scores)
                                    demographies.append({'emotion': {'angry': scores[0], 'disgust': scores[1], 'fear': scores[2], 'happy': scores[3], 'sad': scores[4], 'surprise': scores[5], 'neutral': scores[6]}, 'dominant_emotion': label, 'region': {'x': rect[0], 'y': rect[2], 'w': rect[1], 'h': rect[3]}})

                                else:
                                    demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                            else:
                                demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                        except:  # to avoid exception if no face detected
                            if verbose:
                                print("No face detected")
                            demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                if(model_name == "ResMaskingNet"):
                    faces = detector(frames, cv=False) #LINLIN: BATCH FRAMES
                    #print(time.time()-start)
                    for i in range(fromStorage+1, batch_size):
                        try:
                            box, landmarks, score = faces[i][0]
                            if score > 0.95:
                                rect, face, img = face_detector(box, frames[i])
                                if np.sum([face]) != 0.0:
                                    (
                                        label,
                                        _,
                                        scores,
                                    ) = RMN.detect_emotion_for_single_face_image(face)
                                    #print(scores)
                                    demographies.append({'emotion': {'angry': scores[0], 'disgust': scores[1], 'fear': scores[2], 'happy': scores[3], 'sad': scores[4], 'surprise': scores[5], 'neutral': scores[6]}, 'dominant_emotion': label, 'region': {'x': rect[0], 'y': rect[2], 'w': rect[1], 'h': rect[3]}})

                                else:
                                    demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                            else:
                                demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                        except:  # to avoid exception if no face detected
                            if verbose:
                                print("No face detected")
                            demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                if(model_name == "POSTER_V2-AN7"):
                    faces = detector(frames, cv=False) #LINLIN: BATCH FRAMES
                    tensor_batch = []
                    no_face = []
                    rects = []
                    for i in range(fromStorage+1, batch_size):
                        try:
                            box, _, score = faces[i][0]
                            if score > 0.95:
                                rect, face, _ = face_detector(box, frames[i])
                                rects.append(rect)
                                if np.sum([face]) != 0.0:
                                    no_face.append(False)
                                    with torch.no_grad():
                                        img = Image.fromarray(face)
                                        data = test_preprocess(img)
                                        tensor_batch.append(torch.unsqueeze(data, 0))
                                else:
                                    no_face.append(True)
                            else:
                                no_face.append(True)
                        except IndexError:  # to catch exception when no face detected
                            if verbose:
                                print("No face detected")
                            no_face.append(True)
                            
                    if(tensor_batch):
                        data = torch.cat(tensor_batch, 0)
                        with torch.no_grad():
                            if torch.cuda.is_available():
                                    data.cuda()
                            output = model(data)
                            if torch.cuda.is_available():
                                output = output.cpu()
                        output = output.numpy()
                    
                    index = 0
                    for i in range(0, batch_size-(fromStorage+1)):
                        if(no_face[i]):
                            demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                        else:
                            scores = [output[index][6], output[index][5], output[index][4], output[index][1], output[index][2], output[index][3], output[index][0]]
                            label = FER_CLASSES[np.argmax(scores)]
                            demographies.append({'emotion': {'angry': scores[0], 'disgust': scores[1], 'fear': scores[2], 'happy': scores[3], 'sad': scores[4], 'surprise': scores[5], 'neutral': scores[6]}, 'dominant_emotion': label, 'region': {'x': rects[index][0], 'y': rects[index][2], 'w': rects[index][1], 'h': rects[index][3]}})
                            index = index + 1
                        
                if(model_name == "POSTER_V2-RAF"):
                    faces = detector(frames, cv=False) #LINLIN: BATCH FRAMES
                    tensor_batch = []
                    no_face = []
                    rects = []
                    for i in range(fromStorage+1, batch_size):
                        try:
                            box, _, score = faces[i][0]
                            if score > 0.95:
                                rect, face, _ = face_detector(box, frames[i])
                                rects.append(rect)
                                if np.sum([face]) != 0.0:
                                    no_face.append(False)
                                    img = Image.fromarray(face)
                                    data = test_preprocess(img)
                                    tensor_batch.append(torch.unsqueeze(data, 0))
                                else:
                                    no_face.append(True)
                            else:
                                no_face.append(True)
                        except IndexError:  # to catch exception when no face detected
                            if verbose:
                                print("No face detected")
                            no_face.append(True)
                            
                    if(tensor_batch):
                        data = torch.cat(tensor_batch, 0)
                        with torch.no_grad():
                            if torch.cuda.is_available():
                                    data.cuda()
                            output = model(data)
                            if torch.cuda.is_available():
                                output = output.cpu()
                        output = output.numpy()
                    
                    index = 0
                    for i in range(0, batch_size-(fromStorage+1)):
                        if(no_face[i]):
                            demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                        else:
                            scores = [output[index][5], output[index][2], output[index][1], output[index][3], output[index][4], output[index][0], output[index][6]]
                            label = FER_CLASSES[np.argmax(scores)]
                            demographies.append({'emotion': {'angry': scores[0], 'disgust': scores[1], 'fear': scores[2], 'happy': scores[3], 'sad': scores[4], 'surprise': scores[5], 'neutral': scores[6]}, 'dominant_emotion': label, 'region': {'x': rects[index][0], 'y': rects[index][2], 'w': rects[index][1], 'h': rects[index][3]}})
                            index = index + 1
                            
                if(model_name == "APViT"):
                    faces = detector(frames, cv=False) #LINLIN: BATCH FRAMES
                    tensor_batch = []
                    no_face = []
                    rects = []
                    for i in range(fromStorage+1, batch_size):
                        try:
                            box, _, score = faces[i][0]
                            if score > 0.95:
                                rect, face, _ = face_detector(box, frames[i])
                                rects.append(rect)
                                if np.sum([face]) != 0.0:
                                    no_face.append(False)
                                    data = test_preprocess(dict(img=face))
                                    data['img'] = data['img'][None, ...]
                                    tensor_batch.append(data['img'])
                                else:
                                    no_face.append(True)
                            else:
                                no_face.append(True)
                        except IndexError:  # to catch exception when no face detected
                            if verbose:
                                print("No face detected")
                            no_face.append(True)

                    if(tensor_batch):        
                        data = dict()
                        data['img'] = torch.cat(tensor_batch, 0)
                        with torch.no_grad():
                            if torch.cuda.is_available():
                                    data.cuda()
                            output = classifier(**data, return_loss=False)
                            if torch.cuda.is_available():
                                output = output.cpu()
                    
                    index = 0
                    for i in range(0, batch_size-(fromStorage+1)):
                        if(no_face[i]):
                            demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
                        else:
                            scores = output[index]
                            label = FER_CLASSES[np.argmax(scores)]
                            demographies.append({'emotion': {'angry': scores[0], 'disgust': scores[1], 'fear': scores[2], 'happy': scores[3], 'sad': scores[4], 'surprise': scores[5], 'neutral': scores[6]}, 'dominant_emotion': label, 'region': {'x': rects[index][0], 'y': rects[index][2], 'w': rects[index][1], 'h': rects[index][3]}})
                            index = index + 1

            for i in range(fromStorage+1, len(demographies)):
                demography = demographies[i]
                emotion = demography['emotion']
                region = demography['region']
                x = region['x']
                y = region['y']
                w = region['w']
                h = region['h']
                writer.writerow([x,y,w,h, emotion['angry'], emotion['disgust'], emotion['fear'], emotion['happy'], emotion['sad'], emotion['surprise'], emotion['neutral'], demography['dominant_emotion']])
            
            for i in range(0, len(frames)):

                toc = time.time()
                demography = demographies[i]
                img = frames[i]
                fa = demography["region"]
                if int(fa['w']) > 90:
                    # here, np.uint8 handles showing white area issue
                    # freeze_img = np.zeros(resolution, np.uint8)

                    x = fa['x']
                    y = fa['y']
                    w = fa['w']
                    h = fa['h']

                    cv2.rectangle(
                        img, (x, y), (x+w, y+h), (67, 67, 67), 1
                    )  # draw rectangle to main image

                    # -------------------------------
                    # extract detected face
                    # custom_face = base_img[y : y + h, x : x + w]
                    # -------------------------------
                    # facial attribute analysis

                    if enable_face_analysis == True:

                        if enable_emotion:
                            emotion = demography["emotion"]
                            emotion_df = pd.DataFrame(
                                emotion.items(), columns=["emotion", "score"]
                            )
                            emotion_df = emotion_df.sort_values(
                                by=["score"], ascending=False
                            ).reset_index(drop=True)


                            # background of mood box
                            if(render):
                                # transparency
                                overlay = img.copy()
                                opacity = 0.4

                                if x + w + pivot_img_size < resolution_x:
                                    # right
                                    cv2.rectangle(
                                        img
                                        # , (x+w,y+20)
                                        ,
                                        (x + w, y),
                                        (x + w + pivot_img_size, y + h),
                                        (64, 64, 64),
                                        cv2.FILLED,
                                    )

                                    cv2.addWeighted(
                                        overlay, opacity, img, 1 - opacity, 0, img
                                    )

                                elif x - pivot_img_size > 0:
                                    # left
                                    cv2.rectangle(
                                        img
                                        # , (x-pivot_img_size,y+20)
                                        ,
                                        (x - pivot_img_size, y),
                                        (x, y + h),
                                        (64, 64, 64),
                                        cv2.FILLED,
                                    )

                                    cv2.addWeighted(
                                        overlay, opacity, img, 1 - opacity, 0, img
                                    )

                                for index, instance in emotion_df.iterrows():
                                    current_emotion = instance["emotion"]
                                    emotion_label = f"{current_emotion} "
                                    emotion_score = instance["score"] #/ 100
                                    # print(emotion_score)

                                    bar_x = 35  # this is the size if an emotion is 100%
                                    bar_x = int(bar_x * emotion_score)

                                    if x + w + pivot_img_size < resolution_x:

                                        text_location_y = y + 20 + (index + 1) * 20
                                        text_location_x = x + w

                                        if text_location_y < y + h:
                                            cv2.putText(
                                                img,
                                                emotion_label,
                                                (text_location_x, text_location_y),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (255, 255, 255),
                                                1,
                                            )

                                            cv2.rectangle(
                                                img,
                                                (x + w + 70, y + 13 + (index + 1) * 20),
                                                (
                                                    x + w + 70 + bar_x,
                                                    y + 13 + (index + 1) * 20 + 5,
                                                ),
                                                (255, 255, 255),
                                                cv2.FILLED,
                                            )

                                    elif x - pivot_img_size > 0:

                                        text_location_y = y + 20 + (index + 1) * 20
                                        text_location_x = x - pivot_img_size

                                        if text_location_y <= y + h:
                                            cv2.putText(
                                                img,
                                                emotion_label,
                                                (text_location_x, text_location_y),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (255, 255, 255),
                                                1,
                                            )

                                            cv2.rectangle(
                                                img,
                                                (
                                                    x - pivot_img_size + 70,
                                                    y + 13 + (index + 1) * 20,
                                                ),
                                                (
                                                    x - pivot_img_size + 70 + bar_x,
                                                    y + 13 + (index + 1) * 20 + 5,
                                                ),
                                                (255, 255, 255),
                                                cv2.FILLED,
                                            )

                #cv2.rectangle(freeze_img, (10, 10), (90, 50), (67, 67, 67), -10)
                if(render):
                    cv2.imshow("img", img)

                # if val != 'eof' and audio_frame is not None:
                #     #audio
                #     img, t = audio_frame

            waitTime = int(framewaittime)- int(1000*(time.time()-start) + 0.5)
            if waitTime < 1:
                waitTime = 1
            if cv2.waitKey(waitTime) & 0xFF == ord("q"):  # press q to quit
                f.close()
                break
    
    else:
        print("Video has already been analyzed completely. Results are in ", filename)
    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


models = [
  "VGG-Face",
  "enet_b0_8_best_afew",
  "ResMaskingNet",
  "POSTER_V2-AN7",
  "POSTER_V2-RAF",
  "APViT"
]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]
# 0 - 0
# 1 - 0
# 1 - 4
# 2 - 0
# 2 - 4


if __name__ == "__main__":

    inputfile = ''
    inputdir = ''
    model_int = 0
    backend_int = 0
    batch_size = 1
    render = False
    rtplayback = False
    verbose = False
    

    try:
        opts, args = getopt.getopt(sys.argv[1:],"vrphd:m:b:f:s:",["dir=", "file=", "model=", "backend=", "batch_size="])
    except getopt.GetoptError:
        print ('test.py -d <maindir> -f <videofile> -m <model_number> -b <backend_number> -s <batch_size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -d <maindir> -f <videofile> -m <model_number> -b <backend_number> -s <batch_size> [-r(render), -p(realtime playback), -v(verbose)]')
            sys.exit()
        elif opt in ("-f", "--file"):
            inputfile = arg
        elif opt in ("-d", "--dir"):
            inputdir = arg
        elif opt in ("-m", "--model"):
            model_int = int(arg)
        elif opt in ("-b", "--backend"):
            backend_int = int(arg)
        elif opt in ("-s", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-r"):
            render = True
        elif opt in ("-p"):
            rtplayback = True
        elif opt in ("-v"):
            verbose = True

    if model_int == 1:
        if torch.cuda.is_available():
            fer=HSEmotionRecognizer(model_name=models[model_int],device='cuda')
        else:
            fer=HSEmotionRecognizer(model_name=models[model_int],device='cpu')
    if model_int == 2:
        m = RMN(face_detector=False)
    if model_int == 3:
        
        if torch.cuda.is_available():
            checkpoint = torch.load("POSTER_V2/affectnet-7-model_best.pth")
        else:
            checkpoint = torch.load("POSTER_V2/affectnet-7-model_best.pth", map_location=torch.device('cpu'))
        model = pyramid_trans_expr2(img_size=224, num_classes=7)

        model = torch.nn.DataParallel(model)
        
        if torch.cuda.is_available():
            model.cuda()

        model.load_state_dict(checkpoint['state_dict'])

        model.eval()

        test_preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225]),
                                                            ])
    if model_int == 4:
        
        if torch.cuda.is_available():
            checkpoint = torch.load("POSTER_V2/raf-db-model_best.pth")
        else:
            checkpoint = torch.load("POSTER_V2/raf-db-model_best.pth", map_location=torch.device('cpu'))
        model = pyramid_trans_expr2(img_size=224, num_classes=7)

        model = torch.nn.DataParallel(model)
        
        if torch.cuda.is_available():
            model.cuda()

        model.load_state_dict(checkpoint['state_dict'])

        model.eval()

        test_preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225]),
                                                            ])
    if model_int == 5:

        cfg = mmcv.Config.fromfile("APViT/RAF.py")
        cfg.model.pretrained = None
        cfg.model.extractor.pretrained = None
        cfg.model.vit.pretrained = None

        # build the model and load checkpoint
        classifier = build_classifier(cfg.model)
        load_checkpoint(classifier, "APViT/APViT_RAF-3eeecf7d.pth", map_location='cpu')

        if torch.cuda.is_available():
            classifier = classifier.to("cuda")

        classifier.eval()

        test_preprocess = Compose([
            dict(type='Resize', size=112),
            dict(type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img',])
        ])


    analysis("database", inputdir, inputfile, model_name=models[model_int], detector_backend=backends[backend_int])
    
    #grayscale improve speed by 10%-ish