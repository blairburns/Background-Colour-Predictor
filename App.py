#Import the necessary dependencies
from tkinter import *
from Evaluate import *
from Predict import *
import Predict
import csv

root = Tk()

class Window:
    
    def __init__(self, master):
        self.master = master
        #sets application title
        master.title("Background Colour Predictor")
        #creates window size and is not resizable on either axis
        master.geometry("200x400")
        master.resizable(False, False)
        #creates title label
        Label(master, text="Background Colour Predictor").pack()

        #These functions take the textfield inputs, validates the text is a numeric digit
        #and returns the var as an int, else the var is returned as false.
        def rIint():
            try:
                redInt = int(rI.get())
                print(redInt)
                return redInt
            except:
                print("Not an int")
                return False
        
        def gIint():
            try:
                greenInt = int(gI.get())
                print(greenInt)
                return greenInt
            except:
                print("Not an int")
                return False

        def bIint():
            try:
                blueInt = int(bI.get())
                print(blueInt)
                return blueInt
            except:
                print("Not an int")
                return False
   
        
        
        def setText(): 
            #validate the input data and assign to var
            r = rIint()
            g = gIint()
            b = bIint()
            #Creates a hex value out of the rgb values
            textColor = '#%02x%02x%02x' % (r, g, b)
            #delete the original label
            whiteCanvas.delete(setLabel)
            #sets the Sample Text label to the hex value.
            whiteCanvas.create_text((100, 27), text="Sample Text", fill=textColor)
            
            #normalises the input data for NN to accept
            rN = r /255
            gN = g /255
            bN = b /255

            #Converts the float to str to be written to csv file
            rNs = str(rN)
            gNs = str(gN)
            bNs = str(bN)

            #creates new csv file, writes var to new line and closes workbook
            aa = open('rgbVals.csv','w')
            aa.write(rNs + "\n")
            aa.write(gNs + "\n")
            aa.write(bNs)
            aa.close()
            return textColor
            
        #function called to update evalLabel text to evaluated Accuracy
        def eval(): 
            evalLabel.config(text=evaluate()) #calls function from Evaluate.py

        #function called by predictionBtn()
        def bgColourPredicted():
            #if predicted colour is black, set canvas background to black
            if(predLabel.cget("text") == "Black"):
                predCanvas.config(bg="black")
            else: #else canvas background remains white
                predCanvas.config(bg="white")
            #create a new label with the inputed text colour
            predCanvas.create_text((100, 27), text="Sample Text", fill=setText())

        #called by the predict button
        def predictionBtn():
            #sets label to the predicted colour
            #(prediction is returned as Black or White)
            predLabel.config(text=makePredict())
            bgColourPredicted()


        ###GUI COMPONENTS###

        rI = StringVar()
        gI = StringVar()
        bI = StringVar()
        #create 3 inputs for r, g and b
        redInput = Entry(master, textvariable=rI)
        greenInput = Entry(master, textvariable=gI)
        blueInput = Entry(master, textvariable=bI)
        #add textfields to window
        redInput.pack(fill=X)
        greenInput.pack(fill=X)
        blueInput.pack(fill=X) 

        #create a canvas, grey bg. Add a sample text label
        whiteCanvas = Canvas(master, width=200, height=50, bg="grey")
        whiteCanvas.pack()
        setLabel = whiteCanvas.create_text((100, 27), text="Sample Text")

        #Create the rest of the UI elements
        setBtn = Button(master, text="Set", command=setText)
        accBtn = Button(master, text="Accuracy", command=eval)
        evalLabel = Label(master, width=200, text="--%")
        predBtn = Button(master, text="Predict", command=predictionBtn)
        predLabelStatic = Label(master, width=200, text="Predicted Background Colour:")
        predLabel = Label(master, width=200, text="---")
        predCanvas = Canvas(master, width=200, height=50, bg="white")
        close_button = Button(master, text="Close", command=master.quit)

        #Add the elements to the window
        setBtn.pack()
        accBtn.pack()
        evalLabel.pack()
        predBtn.pack()
        predLabelStatic.pack()
        predLabel.pack()
        predCanvas.pack()
        close_button.pack()
        
#keep the window open on the mainloop
my_gui = Window(root)
root.mainloop()
