# import numpy as np
import os 
import sys

# bounces = {key:val for (key, val) in zip(range(10), [0]*10) }
bounces = [0]*10
starts = [0]*10
visits = [0]*10
totalStarts = 0

with open('site_data.csv') as f:
    for line in f:
        s, e = line.strip("\n").split(",")
        if s=="-1": 
            starts[int(e)] += 1
            totalStarts += 1
        if e=="B": bounces[int(s)] += 1
        if e not in ["B", "C"]: visits[int(e)] += 1

print("Initial distribution:")
for i in range(len(starts)):
    print("Page: " + str(i) + " " + str(starts[i]/totalStarts))

print("Bounce rates:")
for i in range(len(bounces)):
    print("Bounce rate: " + str(i) + " " + str(bounces[i]/visits[i]))
