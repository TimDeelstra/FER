import csv
import sys
import numpy

f = open(sys.argv[1])
reader = csv.reader(f)

f2 = open("Results_grade_file_sorted.csv", "w")
writer = csv.writer(f2)

struct = []

for (l1, l2, l3) in zip(*[reader]*3):
    struct.append((l1, l2, l3))

def sort_id(x):
    return int(x[0][0])

def sort_session(x):
    return int(x[0][1])

def sort_model(x):
    return x[0][2]

struct.sort(key=sort_model)
struct.sort(key=sort_session)
struct.sort(key=sort_id)

#print(struct)
filter = {}
for result in struct:
    print(result)
    if (result[0][0], result[0][1]) in filter:
        filter[(result[0][0], result[0][1])] += 1
    else:
        filter[(result[0][0], result[0][1])] = 1


print(filter)

for(l1, l2, l3) in struct:
    if filter[(l1[0], l1[1])] > 2: 
        writer.writerows([l1, l2, l3])