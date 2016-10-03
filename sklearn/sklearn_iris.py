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


import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors, datasets
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

iris = datasets.load_iris()

X = iris.data
y = iris.target

nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
n_neighbors = 15
clf_nn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance') # 'uniform' or 'distance'
clf_nn.fit(X, y)

clf_adaboost = AdaBoostClassifier(n_estimators=100)
clf_adaboost.fit(X, y)

clf_svm = svm.SVC()
clf_svm.fit(X, y)

# building classification for test scenario
classification = [ [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ], [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ], [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ] ]
for i, c in enumerate(X):
    t = y.item(i)
    for a in range(0,4):
        if (c[a] > classification[t][a][1]):
            classification[t][a][1] = c[a]
        if (classification[t][a][0] == 0 or classification[t][a][0] > c[a]):
            classification[t][a][0] = c[a]

for test_run in range(0,10):
    type = random.randint(0,2)
    sepal_length = random.uniform( classification[type][0][0], classification[type][0][1] )
    sepal_width = random.uniform( classification[type][1][0], classification[type][1][1] )
    petal_length = random.uniform( classification[type][2][0], classification[type][2][1] )
    petal_width = random.uniform( classification[type][3][0], classification[type][3][1] )
    test_arr = [sepal_length, sepal_width, petal_length, petal_width]
    Z_nn = clf_nn.predict([test_arr])
    Z_adaboost = clf_adaboost.predict([test_arr])
    Z_svm = clf_svm.predict([test_arr])
    print ('test run [' + str(test_run) +'] with  test set ' + str(test_arr) + '. should be ['+str(type)+'] and we got: '+ str(Z_nn) + ' / ' + str(Z_adaboost) + ' / ' + str(Z_svm))
