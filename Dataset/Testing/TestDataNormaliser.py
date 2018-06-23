import os
from random import *
import csv

testArrayValues = []
testArrayLabels = []

with open('/Users/blairburns/Documents/DeepLearning/BackgroundColourPredictor/Dataset/Testing/GenTestDataMod500.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        values = []
        colourValues = []

        rVal = row['R']
        rValNorm = float(rVal) / 255
        rValNorm = round(rValNorm, 2)
        values.append(rValNorm)

        gVal = row['G']
        gValNorm = float(gVal) / 255
        gValNorm = round(gValNorm, 2)
        values.append(gValNorm)

        bVal = row['B']
        bValNorm = float(bVal) / 255
        bValNorm = round(bValNorm, 2)
        values.append(bValNorm)

        testArrayValues.append(values)

        colourVal = row['Colour']
        if (colourVal == 'Black'):
            colourVal = [float(1)]
        else:
            colourVal = [float(0)]

        testArrayLabels.append(colourVal)
