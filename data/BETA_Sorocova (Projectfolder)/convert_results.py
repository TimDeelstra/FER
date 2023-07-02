import csv
import sys
from statistics import mean, stdev
import matplotlib.pyplot as plt
import matplotlib
import numpy
from sklearn.metrics import ConfusionMatrixDisplay


variant = ""

filtered = bool(int(sys.argv[2]))
if filtered:
    variant = "_filtered"

f = open(sys.argv[1])
f2 = open(sys.argv[1][:-4] + variant + "_converted.csv", "w+")
f3 = open(sys.argv[1][:-4] + variant + "_confusion.csv", "w+")

reader = csv.reader(f)
writer = csv.writer(f2)
writer_conf = csv.writer(f3)

models = set()

results = {}

detections = {}

for line in reader:
    print("ID:", line[0])
    print("Session:", line[1])
    print("Model:", line[2])
    print("Detections:", line[-1])
    if not((line[0], line[1]) in results):
        results[(line[0], line[1])] = {}

    results[(line[0], line[1])][line[2]] = (line[-1], line[4:-1])
    models.add(line[2])
    if not line[2] in detections:
        detections[line[2]] = []
    detections[line[2]].append(int(line[-1]))

if filtered:
    for result in list(results):
        for model in results[result]:
            detects, _ = results[result][model]
            if int(detects) > 10:
                results.pop(result)
                break

print("Total sessions:", len(results))

models = sorted(models)

model_combo = []
for model in models:
    for model2 in models:
        if model != model2:
            model_combo.append((model, model2))

temp = set()
row_end = []
for (m1, m2) in model_combo:
    if((m1, m2) not in temp and (m2, m1) not in temp):
        row_end.append("OAgree-" + m1 + " with " + m2)
        temp.add((m1, m2))

for (m1, m2) in model_combo:
    row_end.append("TAgree-" + m1 + " with " + m2)
    row_end.append("FAgree-" + m1 + " with " + m2)


writer.writerow(["Id", "Session"] + models + row_end)
writer_conf.writerow(["m1", "m2", "T(m2)-Agree(m1)", "T(m2)-Disagree(m1)", "F(m2)-Agree(m1)", "F(m2)-Disagree(m1)"])

OAgree = {}
TAgree = {}
FAgree = {}

for (m1, m2) in model_combo:
    if (m1, m2) in temp:
        OAgree[(m1, m2)] = []
    TAgree[(m1, m2)] = []
    FAgree[(m1, m2)] = []

for result in dict(sorted(results.items())):
    data = results[result]
    row = [result[0], result[1]]
    
    for model in models:
        (count, detect) = data[model]
        if(detect == []):
            row.append("N/C")
        else:
            row.append(count)
    
    row_end = []
    for (m1, m2) in model_combo:
        _, d1 = data[m1]
        _, d2 = data[m2]
        if(d1 == [] or d2 == []):
            if((m1, m2) in temp):
                row.append("N/A")
            row_end.append("N/A")
            row_end.append("N/A")
        else:
            if((m1, m2) in temp):
                Overal_agree = [e1 == e2 for (e1, e2) in zip(d1,d2)]  
                Overal_agree_percent = len([x for x in Overal_agree if x])/len(Overal_agree)
                row.append(Overal_agree_percent)
                OAgree[(m1, m2)] += (Overal_agree)

            True_agree = [e1 == e2 for (e1, e2) in zip(d1,d2) if e2 == '1']
            if(True_agree == []):
                row_end.append("N/D")
            else:
                True_agree_percent = len([x for x in True_agree if x])/len(True_agree)
                row_end.append(True_agree_percent)
                TAgree[(m1, m2)] += (True_agree)
            
            False_agree = [e1 == e2 for (e1, e2) in zip(d1,d2) if e2 == '0']
            if(False_agree == []):
                row_end.append("N/D")
            else:
                False_agree_percent = len([x for x in False_agree if x])/len(False_agree)
                row_end.append(False_agree_percent)
                FAgree[(m1, m2)] += (False_agree)
    
    writer.writerow(row + row_end)

for (m1, m2) in model_combo:
    if((m1, m2) in temp):
        print("Average Overal Agreement", m1, m2, (len([x for x in OAgree[(m1, m2)] if x])/len(OAgree[(m1, m2)])))
    print("Average True Agreement", m1, m2, (len([x for x in TAgree[(m1, m2)] if x])/len(TAgree[(m1, m2)])))
    print("Average False Agreement", m1, m2, (len([x for x in FAgree[(m1, m2)] if x])/len(FAgree[(m1, m2)])))

for (m1, m2) in model_combo:
    print("confusion matrix", m1, "agreement with", m2)
    a0 = len([x for x in TAgree[(m1, m2)] if x])/len(TAgree[(m1, m2)])
    d0 = 1.0-a0

    a1 = len([x for x in FAgree[(m1, m2)] if x])/len(FAgree[(m1, m2)])
    d1 = 1.0-a1
    print('   ', m2)
    print('  T', ' '*5, 'F')
    print('T', str(a0)[:7],str(d1)[:7])
    print('F', str(d0)[:7], str(a1)[:7])

    writer_conf.writerow([m1,m2,a0,d0,a1,d1])
    disp = ConfusionMatrixDisplay(confusion_matrix=numpy.array([[a0, d0], [d1, a1]]),
                                  display_labels=['True', 'False'])

    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(m1 + " agreement with " + m2 + " segments")
    disp.ax_.set_xlabel(m1)
    disp.ax_.set_ylabel(m2)
    disp.im_.set_norm(matplotlib.colors.Normalize())
    plt.savefig(m1 + "." + m2 + variant + ".png", bbox_inches='tight')

if not(filtered):
    plt.cla()
    plt.clf()
    plt.gca().set(title="Frequency of detections per video")
    colors = plt.cm.Blues([0.3, 0.66, 1.0])
    for index, model in enumerate(models):
        plt.hist(detections[model], bins=50, range=(0, 200), color=colors[index], alpha=0.7, label=model)
    plt.legend()
    plt.xlim([0, 200])
    plt.xlabel("Detections")
    plt.ylabel("Frequency")
    plt.savefig("HistoMulti.png")

    plt.cla()
    plt.clf()
    plt.gca().set(title="Frequency of detections per video")
    colors = plt.cm.Blues([0.3, 0.66, 1.0])
    for index, model in enumerate(models):
        plt.hist(detections[model], bins=50, range=(0, 50), color=colors[index], alpha=0.7, label=model)
    plt.legend()
    plt.xlim([0, 50])
    plt.xlabel("Detections")
    plt.ylabel("Frequency")
    plt.savefig("HistoMultiCrop.png")

    detections_combined = []
    for model in detections:
        detections_combined += detections[model]
    plt.cla()
    plt.clf()
    plt.gca().set(title="Frequency of detections per video")
    colors = plt.cm.Blues([0.3, 0.66, 1.0])
    plt.hist(detections_combined, bins=50, range=(0, 200), color=colors[2])
    plt.xlim([0, 200])
    plt.xlabel("Detections")
    plt.ylabel("Frequency")
    plt.savefig("Histo.png")

    plt.cla()
    plt.clf()
    plt.gca().set(title="Frequency of detections per video")
    colors = plt.cm.Blues([0.3, 0.66, 1.0])
    plt.hist(detections_combined, bins=50, range=(0, 50), color=colors[2])
    plt.xlim([0, 50])
    plt.xlabel("Detections")
    plt.ylabel("Frequency")
    plt.savefig("HistoCrop.png")
    plt.show()

    print("Max", max(detections_combined))
    print("Mean", mean(detections_combined))
    print("StDev", stdev(detections_combined))
