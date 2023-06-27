import csv
import sys
import cv2
import math

#framecount to timestamp
def frames_to_TC (frames, fr):
    h = int(frames / (3600*fr)) 
    m = int(frames / (60*fr)) % 60 
    s = int((frames % (60*fr))/fr) 
    f = int(frames % (60*fr) % fr)
    return ( "%02d:%02d:%02d:%02d" % ( h, m, s, f))

def detect(fcsv, fvid, render=False):
    f = open(fcsv)
    reader = list(csv.reader(f))

    video = cv2.VideoCapture(fvid)
    fps = video.get(cv2.CAP_PROP_FPS)
    fc = video.get(cv2.CAP_PROP_FRAME_COUNT)

    print(fps)
    print(fc)


    if (video.isOpened() == False): 
        print("Error reading video file")

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    
    size = (frame_width, frame_height)

    if(render):
        new_video = fcsv[:-4] + '.avi'
        print("Creating new video containing all Surprise events: ", new_video)

        result = cv2.VideoWriter(new_video, 
                                cv2.VideoWriter_fourcc(*'MPEG'),
                                fps, size)

    print("treshhold: ", (math.ceil(0.2*fps)))
    line_count = 0
    detect_count = 0
    detect_segments = []
    detections = []
    if(fps > 0.0):
        while True:
            try:
                repeat = 0
                detect = False
                for i in range(0, math.ceil(5*fps)):
                    data = reader[line_count + i]
                    _, _, _, _, _, _, _, _, _, _, _, dominant = data
                    if(dominant == "Surprise"):
                        repeat = repeat + 1
                    else:
                        repeat = 0

                    if(repeat > int((math.ceil(0.2*fps))) and not(detect)):
                        print("Surprise detected at frame: ", line_count+i+1, " time: ", frames_to_TC(line_count+i+1, fps))

                        detections.append((detect_count, str(frames_to_TC(line_count+1, fps))))

                        # splice video
                        if(render):
                            video.set(cv2.CAP_PROP_POS_FRAMES, line_count + 1)
                            for i in range(0,(math.ceil(5*fps))):
                                try:
                                    res, frame = video.read()
                                    data = reader[line_count + i]
                                    x, y, w, h, _, _, _, _, _, _, _, _ = data
                                    cv2.rectangle(
                                        frame, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), (67, 67, 67), 1
                                    )
                                    cv2.putText(frame, "detection: " + str(detect_count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2, cv2.LINE_AA)
                                    cv2.putText(frame, "timestamp: " + str(frames_to_TC(line_count+1, fps)), (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2, cv2.LINE_AA)

                                    result.write(frame)
                                except IndexError:
                                    break
                        
                        detect = True
                        detect_count = detect_count + 1

                detect_segments.append(int(detect))  
                line_count = line_count + math.ceil(5*fps)
            except:
                detect_segments.append(int(detect))
                break

        video.release()
        if(render):
            result.release()

    print(detect_count, " surprise events")
    detect_segments.append(detect_count)
    return (detect_segments, detections)

if __name__ == "__main__":
    print(detect(sys.argv[1],sys.argv[2]))
