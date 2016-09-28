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

iris = datasets.load_iris()

X = iris.data
y = iris.target

nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
n_neighbors = 15

for weight in  ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X, y)
    for test_run in range(0,10):
        sepal_length = random.uniform( 4.0, 8.0 )
        sepal_width = random.uniform( 2.0, 5.0 )
        petal_length = random.uniform( 1.0, 6.0 )
        petal_width = random.uniform( 0.1, 3.0 )
        test_arr = [sepal_length, sepal_width, petal_length, petal_width]
        Z = clf.predict([test_arr])
        print (weight + ' test run [' + str(test_run) +'] with a generated test set ' + str(test_arr) + ' we got: '+ str(Z))
