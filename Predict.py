#Import the necessary dependencies
import tensorflow as tf
import numpy as np 
import sys
import csv

#the layers making up the NN
def get_logits(features):
    l1 = tf.layers.dense(features, 3, activation=tf.nn.relu) #3 input nodes, relu act.
    l2 = tf.layers.dense(l1, 4, activation=tf.nn.relu) #1 hidden layer - nodes, relu act.
    l3 = tf.layers.dense(l2, 1, activation=None) #1 output, no act.
    a = l3
    return a

#reads the input data from the csv file set by the app
def readRgb():
    rgbPredictValues = [] #empty array
    with open('rgbVals.csv','r') as bb:
        #for each row in the file, add to array
        reader = csv.reader(bb)
        for row in reader:
            rgbPredictValues.append(float(row[0])) #appends to array
    rgbPredictValuesArray = [] #second empty array
    #array1 is appended to array2, this creates a tensor that the NN can read
    rgbPredictValuesArray.append(rgbPredictValues) 
    print(rgbPredictValuesArray)
    return rgbPredictValuesArray

###for stand alone python script
#predictTest = []
#predictTest.append([0.96, 0.96, 0.96])
###

#the location at which the current model is saved
#The information for the weights and biases
model_dir = './Model/savedModel/' #./

#organises the prediction input data
def get_pred_inputs(feature_data,n_epochs=None, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices( 
        (feature_data))

    dataset = dataset.repeat(n_epochs)
    if shuffle:
        dataset = dataset.shuffle(len(feature_data))
    dataset = dataset.batch(1)
    features = dataset
    return features
    
def pred_model_fn(features, mode):
    a = get_logits(features)
    predictions = tf.greater(a, 0)
    return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions
    )

def predict_input_fn():
    return get_pred_inputs(
    readRgb(), #'predictTest' for stand alone python script
    n_epochs=1,
    shuffle=False
    )

###for GUI ###
def makePredict():
    #create predictor from custom tf.estimator
    predictor = tf.estimator.Estimator(pred_model_fn, model_dir)
    predict = predictor.predict(predict_input_fn) #predict model
    predicted = list(predict)#set predicted tensor as list
    predicted2 = predicted[0]#unwrap first layer of list
    predictedTorF = predicted2[0]#take first element of list
    print(predictedTorF)

    #if predicted is true, background is black
    if (predictedTorF == True): 
        predictedColour = "Black"
    
    #else the background will be white (false)
    if (predictedTorF == False):
        predictedColour = "White"
        
    return predictedColour



###For stand alone python script
#predicter = tf.estimator.Estimator(pred_model_fn, model_dir)
#predict = predicter.predict(predict_input_fn)
#predicted = list(predict)
#predicted2 = predicted[0]
#predictedTorF = predicted2[0]

#if (predictedTorF == True):
#    predictedColour = "Black"
#if (predictedTorF == False):
#    predictedColour = "White"

#print("Background colour prediction: {}".format(predictedColour))
###
    
