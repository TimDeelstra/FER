from detect import detect
import sys
import os
import re
import csv

csv_dir = sys.argv[1]
vid_dir = sys.argv[2]

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
        filter.append(x[len(sys.argv[2]):])

print("CSV files found:", filter)

f = open("data/BETA_Sorocova (Projectfolder)/Results.csv", "w")
writer = csv.writer(f)

for file in filter:
    if(file == "Results.csv"):
        continue
    fcsv = csv_dir + "/" + file
    fvid = vid_dir + "/" + file.split(".")[:-4][0] + ".MP4"
    
    id = re.findall(r"\D(\d{5})\D", file)[0]
    sessie = re.findall(r"Sessie (\d{1})\D", file)[0]
    model = file.split(".")[-3]
    backend = file.split(".")[-2]
    detections = detect(fcsv, fvid)
    writer.writerow([id, sessie, model, backend, detections])
