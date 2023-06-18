import csv
import sys
import cv2

#framecount to timestamp for 25fps
def frames_to_TC (frames):
    h = int(frames / 90000) 
    m = int(frames / 1500) % 60 
    s = int((frames % 1500)/25) 
    f = frames % 1500 % 25
    return ( "%02d:%02d:%02d:%02d" % ( h, m, s, f))

f = open(sys.argv[1])
reader = list(csv.reader(f))

# video = cv2.VideoCapture(sys.argv[2])

# if (video.isOpened() == False): 
#     print("Error reading video file")

# frame_width = int(video.get(3))
# frame_height = int(video.get(4))
   
# size = (frame_width, frame_height)

new_video = sys.argv[1][:-4] + '.avi'
print("Creating new video containing all Surprise events: ", new_video)

# result = cv2.VideoWriter(new_video, 
            #              cv2.VideoWriter_fourcc(*'MJPG'),
            #              10, size)

line_count = 0
detect_count = 0
repeat = 0
while True:
    try:
        data = reader[line_count]
        _, _, _, _, _, _, _, _, _, _, _, dominant = data
        if(dominant == "Surprise"):
            repeat = repeat + 1
        else:
            repeat = 0

        if(repeat == 5):
            print("Surprise detected at frame: ", line_count-1, " time: ", frames_to_TC(line_count-1))
            
            # Todo: splice a video from this detection

            # video.set(cv2.CAP_PROP_POS_FRAMES, line_count - 126)
            for i in range(0,200):
                # res, frame = video.read()
                data = reader[line_count - 127 + i]
                x, w, y, h, _, _, _, _, _, _, _, _ = data
                # cv2.rectangle(
                #     frame, (x, y), (x+w, y+h), (67, 67, 67), 1
                # )

                # result.write(frame)

            detect_count = detect_count + 1


        
        line_count = line_count + 1
    except Exception as e:
        break

# video.release()
# result.release()

print(detect_count, " surprise events")


