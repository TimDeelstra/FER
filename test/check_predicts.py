import csv
import sys

def check(arg1, arg2):
    f1 = open(arg1)
    f2 = open(arg2)

    r1 = csv.reader(f1)
    r2 = csv.reader(f2)

    f1.seek(0)
    end = len(list(r1))
    f1.seek(0)
    f2.seek(0)

    line_counter = 1
    for _ in range(end):
        line = next(r1)
        data = next(r2)

        if line[0] != data[0] or line[1] != data[1] or line[2] != data[2] or line[3] != data[3]:
            print("Non-matching line:", line_counter)
            print(line[:4], data[:4])

        line_counter += 1
    


if __name__ == "__main__":
    print(check(sys.argv[1],sys.argv[2]))