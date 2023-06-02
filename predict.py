import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.commons import functions
import csv
import sys, getopt
from hsemotion.facial_emotions import HSEmotionRecognizer
from batch_face import RetinaFace
#from ffpyplayer.player import MediaPlayer

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

offset = 0
target_size = (224, 224)
detector = RetinaFace(gpu_id=0)
batch_size = 16

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
    db_path,
    dir,
    file,
    model_name="VGG-Face",
    detector_backend="retinaface",
    enable_face_analysis=True,
    frame_threshold=1,
):
    # global variables
    text_color = (255, 255, 255)
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
    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
    start = time.time()

    source = dir + "/" + file

    if(dir==""):
        source = file

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:" + str(fps) + "\n\n")

    rtplayback = False
    render = True

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
        try:
            for d in file.split("/"):
                os.mkdir(path)
                path += "/".join(d)
        except OSError as error: 
            print(error)  
        
        # Open data file in read/write
        try:
            f = open(path + "/" + filename, "a+")
            f.seek(0)
            reader = csv.reader(f)
            writer = csv.writer(f)
        except IOError as e:
            print ("I/O error({0}): {1}".format(e.errno, e.strerror))
        except: #handle other exceptions such as attribute errors
            print ("Unexpected error:", sys.exc_info()[0])

    while(cap.isOpened()):
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
        
        if frames is []:
            break

        print(str(100*cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT)) + "%% completed      ", end="\n")
        #audio_frame, val = player.get_frame()

        # cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

        fromStorage = -1

        demographies = []

        try:
            for img in frames:
                data = next(reader)
                print("frame found:" + str(reader.line_num))
                x, y, w, h, angry, disgust, fear, happy, sad, surprise, neutral, dominant = data
                demography = {'emotion': {'angry': float(angry), 'disgust': float(disgust), 'fear': float(fear), 'happy': float(happy), 'sad': float(sad), 'surprise': float(surprise), 'neutral': float(neutral)}, 'dominant_emotion': dominant, 'region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}}
                demographies.append(demography)
                fromStorage = fromStorage + 1
                print("frame loaded successfully")        
        except StopIteration:
            start = time.time()
            # just extract the regions to highlight in webcam
            #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
                        print("EXCEPTION")
                        demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
            # print(time.time()-start)
            if(model_name == "enet_b0_8_best_afew"):
                faces = detector(frames, cv=False) #LINLIN: BATCH FRAMES
                #print(time.time()-start)
                for i in range(fromStorage+1, batch_size):
                    try:
                        box, landmarks, score = faces[i][0]
                        if score > 0.9:
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
                        print("EXCEPTION")
                        demographies.append({'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}, 'dominant_emotion': "None", 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}})
            
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
                                emotion_score = instance["score"] / 100
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

        if(not(face_detected) and not(fromStorage)):
                writer.writerow([0,0,0,0, 0, 0, 0, 0, 0, 0, 0, "None"])

        face_detected = False
        face_included_frames = 0
        freeze = False
        freezed_frame = 0

        waitTime = int(framewaittime)- int(1000*(time.time()-start) + 0.5)
        if waitTime < 1:
            waitTime = 1
        if cv2.waitKey(waitTime) & 0xFF == ord("q"):  # press q to quit
            f.close()
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


models = [
  "VGG-Face",
  "enet_b0_8_best_afew"
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

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hf:",["dir=", "file=", "model=", "backend="])
    except getopt.GetoptError:
        print ('test.py -d <maindir> -f <videofile> -m <model_number> -b <backend_number>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -d <maindir> -f <videofile> -m <model_number> -b <backend_number>')
            sys.exit()
        elif opt in ("-f", "--file"):
            inputfile = arg
        elif opt in ("-d", "--dir"):
            inputdir = arg
        elif opt in ("-m", "--model"):
            model_int = int(arg)
        elif opt in ("-b", "--backend"):
            backend_int = int(arg)

    if model_int == 1:
        fer=HSEmotionRecognizer(model_name=models[model_int],device='cuda')

    analysis("database", inputdir, inputfile, model_name=models[model_int], detector_backend=backends[backend_int])
    #grayscale improve speed by 10%-ish