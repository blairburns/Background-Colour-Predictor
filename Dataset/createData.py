import os
from random import *
import xlsxwriter

# open a file for writing.
workbook = xlsxwriter.Workbook('GenTrainData.xlsx')
# create the file writer object.
worksheet = workbook.add_worksheet()

#define variables
row = 0
colR = 0
colG = 1
colB = 2
colText = 3
SmplText = "Sample Text"

#repeat for:
for i in range(500):
    #creates value out of 255
    rVal = randint(0, 255)
    gVal = randint(0, 255)
    bVal = randint(0, 255)

    #converts to hex
    hexVal = '#%02x%02x%02x' % (rVal, gVal, bVal)

    #creates formating for Sample text colour
    format = workbook.add_format({'font_color': hexVal })

    worksheet.set_column('D:D', 20)

    #writes the RGB and text values to the spreadsheet
    worksheet.write(row, colR, rVal)
    worksheet.write(row, colG, gVal)
    worksheet.write(row, colB, bVal)
    worksheet.write(row, colText, SmplText, format)

    #moves row to be written to down one
    row += 1

#close the file otherwise might not save
workbook.close()
