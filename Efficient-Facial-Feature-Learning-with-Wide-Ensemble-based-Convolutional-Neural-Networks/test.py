import cv2
from controller import cvision
import time
from model.utils import uimage

source = "../../../Downloads/Proefpersoon51014_Sessie1.MP4"
#source = "media/big_bang.mp4"
if not uimage.initialize_video_capture(source):
    raise RuntimeError("Beep boop")

start = time.time()


while(uimage.is_video_capture_open()):
    print("fps: " + str(1/(time.time()-start)))
    start = time.time()

    img, timestap = uimage.get_frame()
    
   


    fer = cvision.recognize_facial_expression(img, 0, face_detection_method=3, grad_cam=False)

    print(fer.list_emotion)