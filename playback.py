import os
import time
import pandas as pd
import cv2
import csv
import sys, getopt



# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

offset = 0
target_size = (224, 224)


# pylint: disable=too-many-nested-blocks


def playback(
    video,
    datafile,
    ):

    print(video)
    # global variables
    pivot_img_size = 112  # face recognition result image

    enable_emotion = True
    # ------------------------
    # find custom values for this input set
    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    # visualization
    start = time.time()

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    v_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    resolution_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    resolution_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    num_lines = 0
    
    print("Resolution: ", resolution_x, "x", resolution_y)
    print("Framecount: ", str(v_length))
    print("FPS:" + str(fps) + "\n\n")
    

    framewaittime = 1000/fps


    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        exit(0)
    else:
        
        # Open data file in read/write
        try:
            f = open(datafile, "r")
            f.seek(0)
            reader = csv.reader(f)
            num_lines = sum(1 for _ in f)
            f.seek(0)

        except IOError as e:
            print ("I/O error({0}): {1}".format(e.errno, e.strerror))
            print(datafile)
            exit(1)
        except: #handle other exceptions such as attribute errors
            print ("Unexpected error:", sys.exc_info()[0])
            exit(1)

    if(True):
        while(cap.isOpened()):
            if verbose:
                print("fps: " + str(1/(time.time()-start)))
            start = time.time()

            batch_len = 0
            frames = []
            while batch_len < 1:
                _, img = cap.read()

                if img is None:
                    break

                frames.append(img)
                batch_len = batch_len + 1
            
            if frames == []:
                if verbose:
                    print("No more frames, end of file.")
                break
            if verbose:
                print(str(100*cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT)) + "%% completed      ", end="\n")
            else:
                print(str(100*cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT)) + "%% completed      ", end="\r")
            #audio_frame, val = player.get_frame()

            fromStorage = -1
            fromStorageFaces = -1

            faces = []
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
                    fromStorage += 1
                    if verbose:
                        print("frame loaded successfully")  
                    box = [x, w, y, h]
                    faces.append([(box,0,1.0)])
                    fromStorageFaces +=1
            # Run the model for the frames for which we didn't find results from storage   
            except StopIteration:
                pass
                start = time.time()
                # just extract the regions to highlight in webcam
            # Render the face detection and emotion data on the video
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

                    if True:

                        if enable_emotion:
                            emotion = demography["emotion"]
                            emotion_df = pd.DataFrame(
                                emotion.items(), columns=["emotion", "score"]
                            )
                            emotion_df = emotion_df.sort_values(
                                by=["score"], ascending=False
                            ).reset_index(drop=True)


                            # background of mood box
                            if True:
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


if __name__ == "__main__":

    verbose = False
    

    try:
        opts, args = getopt.getopt(sys.argv[1:],"vh")
    except getopt.GetoptError:
        print ('playback.py <video> <datafile> [-v (verbose)]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('playback.py <video> <datafile> [-v (verbose)]')
            sys.exit()
        elif opt in ("-v"):
            verbose = True



    playback(sys.argv[1], sys.argv[2])