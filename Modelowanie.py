import numpy as np #listy i arraye
import os #obsluga systemu i plikow
import sklearn #sci-kit learn
import matplotlib.pyplot as plt
import random
import re #regular expressions

#dataset.csv:
#SystemCodeNumber,Capacity,Occupancy,LastUpdated

filepath = 'd:\\dev\\modelowanie\\dataset2.txt'

datafile = open(filepath, 'r')
lines=datafile.readlines()


Y = []
Parking_Spots = []

for line in lines:

    if line.split(',')[0] not in Parking_Spots:
        Parking_Spots.append(line.split(',')[0])

    if line.split(',')[0] == Parking_Spots[0]:
        Y.append(line.split(',')[2])

    

print(Y)




datafile.close()