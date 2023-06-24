from check_predicts import check
import sys
import os

path = sys.argv[1]
result = set()

for root, dir, files in os.walk(path, followlinks=True):
    for file in files:
        file = os.path.join(root, "".join([x + "." for x in file.split(".")[:-3]]))[:-1]
        if(file[-3:] in ["mp4", "MP4"]):
            result.add(file)

total = []
for video in sorted(result):
    f1 = video + ".POSTER_V2-AN7.retinaface.csv"
    f2 = video + ".POSTER_V2-RAF.retinaface.csv"
    f3 = video + ".APViT.retinaface.csv"

    if(check(f1,f2) != 0 or check(f1,f3) != 0 or check(f2,f3) != 0):
        total.append((video.split("/")[-1],(check(f1,f2), check(f1,f3), check(f2,f3))))
    

print(total)