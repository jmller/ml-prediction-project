#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import config
import data_split


def classifier(X_train, y_train, X_test, x, tune_size, config_list, opt_modulo_params):
    ###
    # Hyperparameterarrays
    ###    
 
    C = [0.5, 1, 10, 100]
    kernel = ['poly', 'rbf']
    degree = [1, 2]
    gamma = [int(x) for x in np.linspace(start = 1, stop = 10, num = 3)]
    
    random_grid = {'C': C, #regularization parameter
           'kernel': kernel,
           'degree': degree, #degree of polynomial function
           'gamma': gamma, #constant of radial basis function
           }
    
    classifier = SVC()    
    
    ###
    # Training and Fitting
    ###    
 
    if x > tune_size+config.min_samples_opt and x%200 == 0 and config_list['optimize_method'] == 1: #Hyperparameter werden alle 10 Ticks optimiert
        if config_list['randomized_search'] == 1:
            grid = RandomizedSearchCV(estimator= classifier, param_distributions=random_grid, n_iter=25, cv = data_split.data_split_CV(X_train), n_jobs=-1)
        else:
            grid = GridSearchCV(estimator= classifier, param_grid=random_grid, cv = data_split.data_split_CV(X_train), n_jobs=-1)
        classifier = grid.fit(X_train, y_train)  
        opt_modulo_params = grid.best_params_
    
    else:
        if opt_modulo_params == 0:
            classifier = SVC(C = 1, degree = 1)    
        else:
            classifier = SVC(**opt_modulo_params)
        classifier.fit(X_train, y_train)    
    y_pred = classifier.predict(np.array([X_test])) #bugfix: https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras
    return y_pred, opt_modulo_params
