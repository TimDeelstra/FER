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

def detect(fcsv, fvid):
    f = open(fcsv)
    reader = list(csv.reader(f))

    video = cv2.VideoCapture(fvid)
    fps = video.get(cv2.CAP_PROP_FPS)
    fc = video.get(cv2.CAP_PROP_FRAME_COUNT)


    if (video.isOpened() == False): 
        print("Error reading video file")

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    
    size = (frame_width, frame_height)

    new_video = fcsv[:-4] + '.avi'
    print("Creating new video containing all Surprise events: ", new_video)

    result = cv2.VideoWriter(new_video, 
                            cv2.VideoWriter_fourcc(*'MPEG'),
                            fps, size)

    print("treshhold: ", int(0.2*(math.ceil(fps))))
    line_count = 0
    detect_count = 0
    repeat = 0
    cooldown = 0
    while True:
        try:
            data = reader[line_count]
            _, _, _, _, _, _, _, _, _, _, _, dominant = data
            if(dominant == "Surprise"):
                repeat = repeat + 1
            else:
                if(cooldown == -1):
                    cooldown = 5*(math.ceil(fps))
                repeat = 0

            if(repeat > int(0.2*(math.ceil(fps))) and cooldown == 0):
                print("Surprise detected at frame: ", line_count-1, " time: ", frames_to_TC(line_count-1, fps))
                
                start = line_count - (math.ceil(fps)*2)
                if(start < 0):
                    start = 0
                # Todo: splice a video from this detection

                video.set(cv2.CAP_PROP_POS_FRAMES, start + 1)
                for i in range(0,(math.ceil(fps)*4)):
                    try:
                        res, frame = video.read()
                        data = reader[start + i]
                        x, y, w, h, _, _, _, _, _, _, _, _ = data
                        cv2.rectangle(
                            frame, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), (67, 67, 67), 1
                        )
                        cv2.putText(frame, str(detect_count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2, cv2.LINE_AA)

                        result.write(frame)
                    except IndexError:
                        break
                
                cooldown = -1    
                detect_count = detect_count + 1

            elif(cooldown > 0):
                cooldown = cooldown - 1 

            
            line_count = line_count + 1
        except:
            break

    video.release()
    result.release()

    print(detect_count, " surprise events")
    return detect_count

if __name__ == "__main__":
    detect(sys.argv[1],sys.argv[2])
