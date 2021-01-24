#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:40:44 2020
"""
import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import config
import data_split

def classifier(X_train, y_train, X_test, x, tune_size, config_list, opt_modulo_params):
    ###
    # Hyperparameterarrays
    ###    
    
    alpha = [float(x) for x in np.linspace(start = 0.05, stop = 1, num = 20)]
    fit_prior = [True, False]
    binarize = [True, False]

    random_grid = {'alpha': alpha,
                   'fit_prior': fit_prior,
                   'binarize': binarize 
               }    
    classifier = BernoulliNB()    
    
    if x > tune_size+config.min_samples_opt and x%200 == 0 and config_list['optimize_method'] == 1:
        if config_list['randomized_search'] == 1:
            grid = RandomizedSearchCV(estimator= classifier, param_distributions=random_grid, cv = data_split.data_split_CV(X_train), n_iter=25, n_jobs=-1)
        else:
            grid = GridSearchCV(estimator= classifier, param_grid=random_grid, cv = data_split.data_split_CV(X_train),  n_jobs=-1) #cross-validation splitter - CV splitter - my_cv --< generator object TimeSeriesSplit.split
        classifier = grid.fit(X_train, y_train)  
        opt_modulo_params = grid.best_params_        
    
    else:
        if opt_modulo_params == 0:
            classifier = BernoulliNB() # default settings of scikit-learn (no settings were given by the paper)
        else:
            classifier = BernoulliNB(**opt_modulo_params)#**grid.best_params_)
        classifier.fit(X_train, y_train)    
    y_pred = classifier.predict(np.array([X_test])) #bugfix: https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras
    return y_pred, opt_modulo_params