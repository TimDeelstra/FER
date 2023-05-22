import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.commons import functions
import csv
import sys
#from ffpyplayer.player import MediaPlayer

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pylint: disable=too-many-nested-blocks


def analysis(
    db_path,
    dir,
    file,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    time_threshold=0,
    frame_threshold=1,
):
    # global variables
    text_color = (255, 255, 255)
    pivot_img_size = 112  # face recognition result image

    enable_emotion = True
    enable_age_gender = False
    # ------------------------
    # find custom values for this input set
    target_size = functions.find_target_size(model_name=model_name)
    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    DeepFace.build_model(model_name=model_name)
    print(f"facial recognition model {model_name} is just built")

    if enable_face_analysis:
        # DeepFace.build_model(model_name="Age")
        # print("Age model is just built")
        # DeepFace.build_model(model_name="Gender")
        # print("Gender model is just built")
        DeepFace.build_model(model_name="Emotion")
        print("Emotion model is just built")
    # -----------------------
    # call a dummy find function for db_path once to create embeddings in the initialization
    DeepFace.find(
        img_path=np.zeros([target_size[0], target_size[1], 3]),
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False,
    )
    # -----------------------
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

    #player = MediaPlayer(source)

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
        print("fps: " + str(1/(time.time()-start)))
        start = time.time()
        _, img = cap.read()

        print(time.time()-start)

        if img is None:
            break

        print(str(100*cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT)) + "%% completed      ", end="\n")
        #audio_frame, val = player.get_frame()

        # cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        raw_img = img.copy()
        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

        fromStorage = False

        try:
            data = next(reader)
            #print(data)
            print("frame found:" + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + "\n")
            x, y, w, h, angry, disgust, fear, happy, sad, surprise, neutral, dominant = data
            faces = [[int(x), int(y), int(w), int(h)]]
            demographies = [{'emotion': {'angry': float(angry), 'disgust': float(disgust), 'fear': float(fear), 'happy': float(happy), 'sad': float(sad), 'surprise': float(surprise), 'neutral': float(neutral)}, 'dominant_emotion': dominant, 'region': {'x': x, 'y': y, 'w': w, 'h': h}}]
            fromStorage = True   
            print("frame loaded successfully")        
        except StopIteration:
            try:
                #start = time.time()
                # just extract the regions to highlight in webcam
                #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                demographies = DeepFace.analyze(
                                img_path=img,
                                detector_backend=detector_backend,
                                enforce_detection=False,
                                silent=True,
                                actions="emotion",
                                align=True
                            )
                # print(time.time()-start)
                faces = []
                for demography in demographies:
                    facial_area = demography["region"]
                    faces.append(
                        (
                            facial_area["x"],
                            facial_area["y"],
                            facial_area["w"],
                            facial_area["h"],
                        )
                    )
            except:  # to avoid exception if no face detected
                faces = []
                demographies = []

            if len(faces) == 0:
                face_included_frames = 0

        detected_faces = []
        face_index = 0
    
        print(time.time()-start)
        for x, y, w, h in faces:
            if w > 90:  # discard small detected faces

                face_detected = True
                if face_index == 0:
                    face_included_frames = (
                        face_included_frames + 1
                    )  # increase frame for a single face

                # if(render):
                #     cv2.rectangle(
                #         img, (x, y), (x + w, y + h), (67, 67, 67), 1
                #     )  # draw rectangle to main image

                #     cv2.putText(
                #         img,
                #         str(frame_threshold - face_included_frames),
                #         (int(x + w / 4), int(y + h / 1.5)),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         4,
                #         (255, 255, 255),
                #         2,
                #     )

                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]  # crop detected face

                # -------------------------------------

                detected_faces.append((x, y, w, h))
                face_index = face_index + 1

                # -------------------------------------

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            freeze = True
            # base_img = img.copy()
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()

        if freeze == True:

            toc = time.time()

            if freezed_frame == 0:
                freeze_img = base_img.copy()
                # here, np.uint8 handles showing white area issue
                # freeze_img = np.zeros(resolution, np.uint8)

                for detected_face in detected_faces_final:
                    x = detected_face[0]
                    y = detected_face[1]
                    w = detected_face[2]
                    h = detected_face[3]

                    cv2.rectangle(
                        freeze_img, (x, y), (x + w, y + h), (67, 67, 67), 1
                    )  # draw rectangle to main image

                    # -------------------------------
                    # extract detected face
                    # custom_face = base_img[y : y + h, x : x + w]
                    # -------------------------------
                    # facial attribute analysis

                    if enable_face_analysis == True:

                        if len(demographies) > 0:
                            # directly access 1st face cos img is extracted already
                            demography = demographies[0]

                            if enable_emotion:
                                emotion = demography["emotion"]
                                emotion_df = pd.DataFrame(
                                    emotion.items(), columns=["emotion", "score"]
                                )
                                emotion_df = emotion_df.sort_values(
                                    by=["score"], ascending=False
                                ).reset_index(drop=True)

                                # store emotion data
                                if(not(fromStorage)):
                                    writer.writerow([x,y,w,h, emotion['angry'], emotion['disgust'], emotion['fear'], emotion['happy'], emotion['sad'], emotion['surprise'], emotion['neutral'], demography['dominant_emotion']])

                                # background of mood box
                                if(render):
                                    # transparency
                                    overlay = freeze_img.copy()
                                    opacity = 0.4

                                    if x + w + pivot_img_size < resolution_x:
                                        # right
                                        cv2.rectangle(
                                            freeze_img
                                            # , (x+w,y+20)
                                            ,
                                            (x + w, y),
                                            (x + w + pivot_img_size, y + h),
                                            (64, 64, 64),
                                            cv2.FILLED,
                                        )

                                        cv2.addWeighted(
                                            overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img
                                        )

                                    elif x - pivot_img_size > 0:
                                        # left
                                        cv2.rectangle(
                                            freeze_img
                                            # , (x-pivot_img_size,y+20)
                                            ,
                                            (x - pivot_img_size, y),
                                            (x, y + h),
                                            (64, 64, 64),
                                            cv2.FILLED,
                                        )

                                        cv2.addWeighted(
                                            overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img
                                        )

                                    for index, instance in emotion_df.iterrows():
                                        current_emotion = instance["emotion"]
                                        emotion_label = f"{current_emotion} "
                                        emotion_score = instance["score"] / 100

                                        bar_x = 35  # this is the size if an emotion is 100%
                                        bar_x = int(bar_x * emotion_score)

                                        if x + w + pivot_img_size < resolution_x:

                                            text_location_y = y + 20 + (index + 1) * 20
                                            text_location_x = x + w

                                            if text_location_y < y + h:
                                                cv2.putText(
                                                    freeze_img,
                                                    emotion_label,
                                                    (text_location_x, text_location_y),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,
                                                    (255, 255, 255),
                                                    1,
                                                )

                                                cv2.rectangle(
                                                    freeze_img,
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
                                                    freeze_img,
                                                    emotion_label,
                                                    (text_location_x, text_location_y),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,
                                                    (255, 255, 255),
                                                    1,
                                                )

                                                cv2.rectangle(
                                                    freeze_img,
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
                cv2.imshow("img", freeze_img)

            # if val != 'eof' and audio_frame is not None:
            #     #audio
            #     img, t = audio_frame

        elif(render):

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
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
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




analysis("database", "../..", "Downloads/Proefpersoon11001_sessie1.MP4", model_name=models[0], detector_backend=backends[0])
#grayscale improve speed by 10%-ish

