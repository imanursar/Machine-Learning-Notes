# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 11:59:05 2018

@author: imanursar
"""

import numpy as np
from sklearn import datasets
from sklearn.learning_curve import validation_curve
from sklearn.ensemble import RandomForestClassifier

digits = datasets.load_digits()
X,y = digits.data, digits.target
series = [10, 25, 50, 100, 150, 200, 250, 300]
RF = RandomForestClassifier(random_state=101)
train_scores, test_scores = validation_curve(RF,
                                             X, y, 'n_estimators', param_range=series,
                                             cv=10, scoring='accuracy',n_jobs=-1)