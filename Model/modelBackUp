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
displayStep = 1

#x placeholder is the numberof inputs
#y placeholder is the number of outputs
x = tf.placeholder("float", [None, 3]) #None, 3
y = tf.placeholder("float", [None, 2])
#first values


#layer 1
w1 = tf.Variable(tf.truncated_normal([3, 4], stddev=0.1)) #No. of inputs and No. of nodes in layer 1 #, stddev=0.1 truncated_normal
b1 = tf.Variable(tf.zeros([4])) #No. of nodes in layer 1
y1 = tf.matmul(x, w1) + b1 #Takes placeholder values #tanh

#layer 2
w2 = tf.Variable(tf.truncated_normal([4, 2], stddev=0.1)) # 4 inputs and 2 outputs ##[4,2]
b2 = tf.Variable(tf.zeros([2])) #2
y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2) #sigmoid
#y2 = tf.matmul(y1, w2) + b2

#w3 = tf.Variable(tf.truncated_normal([4, 2], stddev=0.1)) # 4 inputs and 2 outputs
#b3 = tf.Variable(tf.zeros([2]))
#y3 = tf.nn.sigmoid(tf.matmul(y2, w3) + b3) #sigmoid


#output
#wO = tf.Variable(tf.truncated_normal([2, 2], stddev=0.1))
#bO = tf.Variable(tf.zeros([2]))
a = y2#tf.nn.softmax(tf.matmul(y2, wO) + bO) y2
a_ = tf.placeholder("float", [None, 2]) #softmax


#cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a)))
#cross_entropy = -tf.reduce_sum(y*tf.log(a))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

################


#training

init = tf.global_variables_initializer() #initialises tensorflow

with tf.Session() as sess:
    sess.run(init) #runs the initialiser

    writer = tf.summary.FileWriter("/Users/blairburns/Documents/DeepLearning/BackgroundColourPredictor/Logs")
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    for iteration in range(trainingIteration):
        avg_cost = 0
        totalBatch = int(len(trainArrayValues)/batchSize) #1000/100
        #totalBatch = 10

        for i in range(batchSize):
            start = i
            end = i + batchSize #100

            xBatch = trainArrayValues[start:end]
            yBatch = trainArrayLabels[start:end]

            #feeding training data

            sess.run(optimizer, feed_dict={x: xBatch, y: yBatch})

            i += batchSize

            avg_cost += sess.run(cross_entropy, feed_dict={x: xBatch, y: yBatch})/totalBatch

            if iteration % displayStep == 0:
                print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

        #
    print("Training complete")


    predictions = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: testArrayValues, y: testArrayLabels}))


#iterations




#accuracy

#with tf.Session() as sess:
#    sess.run(init)

#    predictions = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))

#    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
#    xtest = xTestT.eval() ## converts tensor back to array
#    ytest = yTestT.eval() ## converts tensor back to array
#    print("Accuracy:", accuracy.eval({x: testArrayValues, y: testArrayLabels}))





#prediction
