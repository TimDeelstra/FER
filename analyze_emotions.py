import csv
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay

csv_dir = sys.argv[1]

f = open("data/BETA_Sorocova (Projectfolder)/Results_all_emotions.csv", "w")
writer = csv.writer(f)

results = set()
if (csv_dir[-1] == "/"):
    csv_dir = csv_dir[:-1]

models = set()
backends = set()

emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]



for root, dir, files in os.walk(csv_dir, followlinks=True):
    for file in files:
        if(file.split('.')[-2] != "retinaface" or file.split(".")[-3] in ['MP4', 'mp4']):
            continue
    
        models.add(file.split(".")[-3])
        backends.add(file.split(".")[-2])
        
        results.add(os.path.join(root, ".".join(file.split(".")[:-3])))

emo_total = {}

for model in models:
    emo_total[model] = {}
    for emotion in emotions:
        emo_total[model][emotion] = 0

agreement = {}

for model in models:
    agreement[model] = {}
    for model2 in models:
        if (model == model2):
            continue
        agreement[model][model2] = {}
        for emo in emotions:
            agreement[model][model2][emo] = {}
            for emo2 in emotions:
                agreement[model][model2][emo][emo2] = 0

for count, result in enumerate(results):
    for model in models:
        for backend in backends:
            file_name = ".".join([result, model, backend, "csv"])

            for model2 in models:
                if model == model2:
                    continue
                f1 = open(file_name)
                f2 = open(".".join([result, model2, backend, "csv"]))

                r1 = csv.reader(f1)
                r2 = csv.reader(f2)

                l1 = list(r1)
                l2 = list(r2)
                for index, line in enumerate(l1):
                    emo = line[-1]
                    emo2 = l2[index][-1]

                    # temp fix for incorrect APViT data labeling
                    # if model == "APViT":
                    #     if emo == "Sadness":
                    #         emo = "Happiness"
                    #     elif emo == "Happiness":
                    #         emo = "Sadness"

                    # if model2 == "APViT":
                    #     if emo2 == "Sadness":
                    #         emo2 = "Happiness"
                    #     elif emo2 == "Happiness":
                    #         emo2 = "Sadness"

                    if emo == "None":
                        continue
                    agreement[model][model2][emo][emo2] += 1
                    emo_total[model][emo] += 1

    print((count+1)/92, end="\r")

print(agreement)
for model in models:
    for model2 in models:
        matrix = []
        if (model == model2):
            continue
        for backend in backends:
            print(model, model2)
            line = []
            total_overall = 0
            agree = 0
            for emo in emotions:
                agree += agreement[model][model2][emo][emo]
                total = sum(agreement[model][model2][emo].values())
                subline = []
                for emo2 in emotions:
                    total_overall += agreement[model][model2][emo][emo2]
                    subline.append(agreement[model][model2][emo][emo2]/total)
                    
                print(subline)
                matrix.append(subline)
                line + subline        

            line = [model, model2, backend, (agree/total_overall)] + line
            writer.writerow(line)
            print()
            print("Overall agreement:", (agree/total_overall)) 

            disp = ConfusionMatrixDisplay(confusion_matrix=np.array(matrix), display_labels=["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])

            disp.plot(cmap=plt.cm.Blues)
            disp.ax_.set_title(model2 + " agreement with " + model + " classifications")
            disp.ax_.set_xlabel(model2)
            disp.ax_.set_ylabel(model)
            disp.im_.set_norm(matplotlib.colors.Normalize())
            plt.savefig(model + "." + model2 + ".emotions.png", bbox_inches='tight')

writer.writerow([])

for model in models:
    total = sum(emo_total[model].values())//2
    print(model)
    for emo in emotions:
        print(emo, emo_total[model][emo]//2, ((emo_total[model][emo]//2) / total))
        writer.writerow([model, emo, emo_total[model][emo]//2])

    print()

