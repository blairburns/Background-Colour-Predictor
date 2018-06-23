import tensorflow as tf
import numpy as np

#import training data
#import test data

#test VARIABLES
xTest = [0.89, 0.76, 0.98]#[0.89],[0.76],[0.98]
xTestT = tf.reshape(xTest, [1,3])

print(tf.Session().run(xTestT))

with tf.Session() as sess:
    b = xTestT.eval()
    print(b)
