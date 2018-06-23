import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '/Users/blairburns/Documents/DeepLearning/BackgroundColourPredictor/Dataset/Testing/')
sys.path.insert(0, '/Users/blairburns/Documents/DeepLearning/BackgroundColourPredictor/Dataset/Training/')
#other files
from TestDataNormaliser import *
from TrainDataNormaliser import *

batchSize = 1

def get_logits(features):
    l1 = tf.layers.dense(features, 3, activation=tf.nn.relu)
    l2 = tf.layers.dense(l1, 4, activation=tf.nn.relu)
    l3 = tf.layers.dense(l2, 1, activation=None)
    a = l3
    return a

def get_loss(a, labels):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=a, labels=labels)

def get_train_op(loss):
    learning_rate = 1e-3
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    return optimizer.minimize(loss, global_step=tf.train.get_global_step())

def get_inputs(feature_data, label_data, batch_size, n_epochs=None, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices( #from_tensor_slices
        (feature_data, label_data))

    dataset = dataset.repeat(n_epochs)
    if shuffle:
        dataset = dataset.shuffle(len(feature_data))
    dataset = dataset.batch(batch_size)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

def model_fn(features, labels, mode):
    a = get_logits(features)
    loss = get_loss(a, labels)
    train_op = get_train_op(loss)
    predictions = tf.greater(a, 0)
    accuracy = tf.metrics.accuracy(labels, predictions)
    return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops={'Accuracy': accuracy},
    predictions=predictions
    )

def eval_input_fn():
    return get_inputs(
    testArrayValues,
    testArrayLabels,
    batchSize,
    n_epochs=1,
    shuffle=False
    )

model_dir = '/Model/savedModel'

def evaluate(): #For GUI
    estimator = tf.estimator.Estimator(model_fn, model_dir)
    acc = estimator.evaluate(eval_input_fn)
    print("Determining Accuracy...")
    print("Accuracy: {0:.2%}".format(acc["Accuracy"]), "Loss: ", (acc["loss"]))
    accAcc = "{0:.2%}".format(acc["Accuracy"])
    return accAcc

#For stand alone python script

#estimator = tf.estimator.Estimator(model_fn, model_dir)
#acc = estimator.evaluate(eval_input_fn)
#print("Determining Accuracy...")
#print("Accuracy: {0:.2%}".format(acc["Accuracy"]), "Loss: ", (acc["loss"]))
