#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2016 Martin Kauss (yo@bishoph.org)

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""


import sys
import tensorflow as tf
import numpy as np
import random

label_one_hot = {'Iris-setosa': [1,0,0], 'Iris-versicolor': [0,1,0], 'Iris-virginica': [0,0,1]}
result_2_label = {0: 'no idea', 1: 'Iris-setosa', 2: 'Iris-versicolor', 3: 'Iris-virginica'}

def get_data_from_csv(filename):
    data = [ ]
    labels = [ ]
    for line in file(filename):
        row = line.split(",")
        temp = [ ]
        for x in range(0,4):
            temp.append(float(row[x]))
        label = row[4]
        label = label.replace('\n', '')
        if (label in label_one_hot):
            labels.append(label_one_hot[label])
        data.append(temp)
    return data, labels

print ('loading and extracting data ...'),
x_input, y_input = get_data_from_csv('data/iris.data.csv')
print ('done.')

print ('initializing placeholders and variables ...'),
# placeholders and variables
x=tf.placeholder(tf.float32,shape=[None,4]) # input
y_=tf.placeholder(tf.float32,shape=[None, 3]) # output

# weight and bias
W=tf.Variable(tf.zeros([4,3]))
b=tf.Variable(tf.zeros([3]))

# softmax classification
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

# optimizer
train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

# accuracy 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print ('done.')

print ('initialing session...'),
# session parameters
session = tf.InteractiveSession()

# initialising variables
init = tf.initialize_all_variables()
session.run(init)
print ('done.')

print ('learning ...')
# number of interations
epoch=100

for step in xrange(epoch):
    _, c = session.run([train_step, cross_entropy], feed_dict={x: x_input, y_: y_input})
    sys.stdout.write("\r%d" % step)
    sys.stdout.flush()
print ('\ndone.')

print ('running test set...')
for test_run in range(0,10):

    sepal_length = random.uniform( 4.0, 8.0 )
    sepal_width = random.uniform( 2.0, 5.0 )
    petal_length = random.uniform( 1.0, 6.0 )
    petal_width = random.uniform( 0.1, 3.0 )

    test_arr = np.asarray([ sepal_length, sepal_width, petal_length, petal_width ])
    test_set = test_arr.reshape(1,4)

    probabilities = y.eval(feed_dict={x: test_set}, session=session)
    predictions = session.run(tf.arg_max(y, 1), feed_dict={x: test_set})

    print ('test run [' + str(test_run) +'] with a generated test set ' + str(test_arr) + ' we got: ')
    print ('probabilities: ' + str(probabilities))
    print ('prediction: ' + str(predictions[0]) + ' == ' + result_2_label[predictions[0]+1])
    print ('                        ')

session.close()
