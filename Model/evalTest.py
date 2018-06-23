import tensorflow as tf
import numpy as np
import sys
#sys.path.insert(0, '/Users/blairburns/Documents/DeepLearning/BackgroundColourPredictor/Model')

#from model import *
def get_logits(features):
    l1 = tf.layers.dense(features, 3, activation=tf.nn.relu)
    l2 = tf.layers.dense(l1, 4, activation=tf.nn.relu)
    l3 = tf.layers.dense(l2, 1, activation=None)
    a = l3
    return a


predictTest = []
predictTest.append([0.96, 0.96, 0.96])

model_dir = './savedModel'

def get_pred_inputs(feature_data,n_epochs=None, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices( #from_tensor_slices
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
    predictTest,
    n_epochs=1,
    shuffle=False
    )

predicter = tf.estimator.Estimator(pred_model_fn, model_dir)
predict = predicter.predict(predict_input_fn)
predicted = list(predict)
predicted2 = predicted[0]
predictedTorF = predicted2[0]

if (predictedTorF == True):
    predictedColour = "Black"
if (predictedTorF == False):
    predictedColour = "White"

print("Background colour prediction: {}".format(predictedColour))
