import os
from random import *
import csv

#create train arrays
trainArrayValues = []
trainArrayLabels = []

#read the training data
with open('/Users/blairburns/Documents/DeepLearning/BackgroundColourPredictor/Dataset/Training/TraingDataMod1000.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        #arrays which will be reset each time called
        values = []
        colourValues = []

        #take r value, normalise, round to 2dp and append to values
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
        #print(rValNorm, gValNorm, bValNorm)

        #append values to trainArrayValues
        trainArrayValues.append(values)

        #if the colour row value is Black, colourVal is 1
        colourVal = row['Colour']
        if (colourVal == 'Black'):
            colourVal = [float(1)]
        else:
            colourVal = [float(0)]

        trainArrayLabels.append(colourVal)
