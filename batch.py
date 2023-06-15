import csv
import sys
import os
import subprocess

f = open(sys.argv[1])
reader = csv.reader(f)
next(reader)

ids = set()

result = []

for root, dir, files in os.walk(sys.argv[2], followlinks=True):
    for file in files:
        result.append(os.path.join(root, file))

filter = []
for x in result:
    if(x[-3:] in ["mp4", "MP4"]):
        filter.append(x[len(sys.argv[2])+1:])

print("MP4 files found:")
print(filter)

while True:
    try:
        data = next(reader)
        id, participant_id, timestamp, session, predicted_error_type = data
        if(int(session) < 3 and predicted_error_type == "robot"):
            f1 = [i for i in filter if str(participant_id) in i]
            f2 = [i for i in f1 if "Sessie1" in i or "sessie1" in i]
            # f2 = filter

            for m in [3,4,5]:
                for file in f2:
                    print(["python", "predict.py", "-m" , str(m), "-b", "4", "-d", sys.argv[2], "-f", file, "-s", "8"])
                    subprocess.run(["python", "predict.py", "-m" , str(m), "-b", "4", "-d", sys.argv[2], "-f", file, "-s", "8"])
            ids.add((participant_id, session))
    except:
        break

print("Total count of robot errors: ", len(ids))