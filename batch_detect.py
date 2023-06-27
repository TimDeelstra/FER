from detect import detect
import sys
import os
import re
import csv
import numpy as np

csv_dir = sys.argv[1]
vid_dir = sys.argv[2]
render = sys.argv[3]

result = []
if (csv_dir[-1] == "/"):
    csv_dir = csv_dir[:-1]

if (vid_dir[-1] == "/"):
    vid_dir = vid_dir[:-1]

for root, dir, files in os.walk(csv_dir, followlinks=True):
    for file in files:
        result.append(os.path.join(root, file))

filter = []
for x in result:
    if(x[-3:] in ["csv"]):
        filter.append(x[len(sys.argv[1]):])

print("CSV files found:", filter)

f = open("/mnt/v/ownCloud - Tim Deelstra@vu.data.surfsara.nl/Sorocova - Facial Expression Recognition/BETA_Sorocova (Projectfolder)/Results.csv", "w")
f2 = open("/mnt/v/ownCloud - Tim Deelstra@vu.data.surfsara.nl/Sorocova - Facial Expression Recognition/BETA_Sorocova (Projectfolder)/Results_grade_file.csv", "w")
writer = csv.writer(f)
grade_writer = csv.writer(f2)

for file in filter:
    if(file.split('.')[-2] != "retinaface" or file.split(".")[-3] in ['MP4', 'mp4']):
        continue
    fcsv = csv_dir + "/" + file
    fvid = vid_dir + "/" + file.split(".")[:-4][0] + ".MP4"
    print(fcsv, fvid)
    
    id = re.findall(r"\D(\d{5})\D", file)[0]
    sessie = re.findall(r"Sessie (\d{1})\D", file)[0]
    model = file.split(".")[-3]
    backend = file.split(".")[-2]
    (detect_segments, detections) = detect(fcsv, fvid, render = bool(int(render)))
    writer.writerow([id, sessie, model, backend] + detect_segments)
    if(len(detections) <= 10 and len(detections) > 0):
        detections = np.array(detections)
        grade_writer.writerows([[id, sessie, model, 'Section'] + list(detections[:,0]), ['', '', '', 'Timestamp'] + list(detections[:,1]), ['', '', '', 'Grade'] + len(detections)*['']])

f.close()
