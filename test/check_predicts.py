import csv
import sys

f1 = open(sys.argv[1])
f2 = open(sys.argv[2])

r1 = csv.reader(f1)
r2 = csv.reader(f2)

counter = 1
for line in r1:
    data = next(r2)

    if line[0] != data[0] or line[1] != data[1] or line[2] != data[2] or line[3] != data[3]:
        print("Non-matching line:", counter)
        print(line[:4], data[:4])

    counter += 1