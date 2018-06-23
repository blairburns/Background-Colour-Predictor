import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '/Users/blairburns/Documents/DeepLearning/BackgroundColourPredictor/Dataset/Testing/')
sys.path.insert(0, '/Users/blairburns/Documents/DeepLearning/BackgroundColourPredictor/Dataset/Training/')
#other files
from TestDataNormaliser import *
from TrainDataNormaliser import *

learning_rate = 0.01
trainingIteration = 10
batchSize = 100
displayStep = 2
max_steps = 5

x = tf.placeholder("float", [None, 3])
y = tf.placeholder("float", [None, 2])

def get_logits(features):

    x = tf.layers.dense(features, 4, activation=tf.nn.relu)

    x = tf.layers.dense(x, 4, activation=tf.nn.relu)


    logits = tf.layers.dense(x, 3, activation=None)
    return logits


def get_loss(logits, labels):
    """tf.nn.sigmoid_cross_entropy_with_logits is numerically stable."""
    # #cost function
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a)))
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=labels)


def get_train_op(loss):
    """There are better options than standard SGD. Try the following."""
    learning_rate = 1e-3
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer.minimize(loss)


def get_inputs(feature_data, label_data, batch_size, n_epochs=None,
               shuffle=True):
    """
    Get features and labels for training/evaluation.

    Args:
        feature_data: numpy array of feature data.
        label_data: numpy array of label data
        batch_size: size of batch to be returned
        n_epochs: number of epochs to train for. None will result in repeating
            forever/until stopped
        shuffle: bool flag indicating whether or not to shuffle.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (feature_data, label_data))

    dataset = dataset.repeat(n_epochs)
    if shuffle:
        dataset = dataset.shuffle(len(feature_data))
    dataset = dataset.batch(batch_size)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels




def model_fn(features, labels, mode):
    logits = get_logits(features)
    loss = get_loss(logits, labels)
    train_op = get_train_op(loss)
    predictions = tf.greater(logits, 0)
    accuracy = tf.metrics.accuracy(labels, predictions)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op,
        eval_metric_ops={'accuracy': accuracy}, predictions=predictions)

def train_input_fn():
    return get_inputs(trainArrayValues, trainArrayLabels, batchSize)


def eval_input_fn():
    return get_inputs(
        testArrayValues, testArrayLabels, batchSize, n_epochs=1, shuffle=False)

# Where variables and summaries will be saved to
model_dir = './savedModel'

estimator = tf.estimator.Estimator(model_fn, model_dir)
estimator.train(train_input_fn, max_steps=max_steps)

estimator.evaluate(eval_input_fn)
