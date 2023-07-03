import csv
import sys
import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay

f = open(sys.argv[1])
reader = csv.reader(f)

struct = {}

videos = set()
models = set()

for (l1, l2, l3) in zip(*[reader]*3):
    if (l1[0], l1[1]) not in struct:
        struct[(l1[0],l1[1])] = {}
    
    eval = []
    for x in l1[4:]:
        if(x != ''):
            eval.append((l2[4 + int(x)],l3[4 + int(x)]))

    struct[(l1[0],l1[1])][l1[2]] = eval
    videos.add((l1[0], l1[1]))
    models.add(l1[2])

print("Total Videos", len(videos))


result = {}

total_fpositive = set()
total_tpositive = set()
total = set()

for model in models:
    result[model] = {'TP':0,'FP':0,'FN': set(), 'total':0}

for model in models:
    for (id, ses) in videos:
        if model in struct[(id, ses)]:
            
            data = struct[(id, ses)][model]

            result[model]['total'] += len(data)
            for eval in data:
                total.add((id, ses, eval[0]))
                if(eval[1] == 'T'):
                    result[model]['TP'] += 1
                    total_tpositive.add((id, ses, eval[0]))
                    for model2 in models:
                        if model2 != model:
                            if model2 in struct[(id, ses)]:
                                
                                data2 = struct[(id, ses)][model2]
                                found = False
                                for eval2 in data2:
                                    if eval[0] == eval2[0]:
                                        found = True
                                        break
                                if not(found):
                                    result[model2]['FN'].add((id, ses, eval[0]))
                            else:
                                result[model2]['FN'].add((id, ses, eval[0]))
                else:
                    result[model]['FP'] += 1
                    total_fpositive.add((id, ses, eval[0]))

      
print("total true positive", len(total_tpositive))
print("total false positive", len(total_fpositive))
print("total positives", len(total))

for model in result:
    data = result[model]
    print(model, data['TP'], data['FP'], len(data['FN']), data['total'])
    
