import csv
import sys
import os
import subprocess

f = open(sys.argv[1])
reader = csv.reader(f)
next(reader)

ids = set()

result = []
path = sys.argv[2]
if (path[-1] == "/"):
    path = path[:-1]

for root, dir, files in os.walk(path, followlinks=True):
    for file in files:
        result.append(os.path.join(root, file))

filter = []
for x in result:
    if(x[-3:] in ["mp4", "MP4"]):
        filter.append(x[len(sys.argv[2]):])

print("MP4 files found:")
print(filter)

while True:
    try:
        data = next(reader)
        id, participant_id, timestamp, session, predicted_error_type = data
        if(not((participant_id, session) in ids) and int(session) < 3 and predicted_error_type == "robot"):
            f1 = [i for i in filter if str(participant_id) in i]
            f2 = [i for i in f1 if "Sessie"+str(session) in i or "sessie"+str(session) in i]
            print(f2)
            # f2 = filter

            for m in [3,4,5]:
                for file in f2:
                    print(["python", "predict.py", "-m" , str(m), "-b", "4", "-d", sys.argv[2], "-f", file, "-s", "12"])
                    subprocess.run(["python", "predict.py", "-m" , str(m), "-b", "4", "-d", sys.argv[2], "-f", file, "-s", "12"])
            ids.add((participant_id, session))
    except:
        break

subprocess.run(["git pull"])
subprocess.run(["git", "add", "data/\*.csv"])
subprocess.run(["git", "commit", "-m", "\"Added new data from batch execution\""])
subprocess.run(["git", "push"])
print("Total count of robot errors processed: ", len(ids))