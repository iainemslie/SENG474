#!/usr/bin/env python3
import argparse
import math

def main():
    infile  = open("result.txt", "r")
    result = infile.read()

    infile  = open("trainlabels.txt", "r")
    test_labels = infile.read()

    num_diff = 0
    num_total = 0
    r = result.splitlines()
    t = test_labels.splitlines()
    for num in range(len(r)):
        num_total = num_total + 1
        if r[num] != t[num]:
            num_diff = num_diff + 1

    print("Total: "+str(num_total))
    print("Different: "+ str(num_diff))
    print("Accuracy: " + str(1-int(num_diff)/int(num_total)))

if __name__ == "__main__":
    main()
