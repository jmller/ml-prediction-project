#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test/Train Split
"""
import numpy as np

import discrete_layer as dl
from sklearn.model_selection import TimeSeriesSplit

import config

def data_split(X, y, x, timeseries, config_list):
    
    # Waits for init of all technical_indicators:
    tune_size = np.max(X[:].isna().sum())
    if config_list['expanding_window_ml'] == 1 or x < config.window_size+tune_size: # expanding window
        X_train = X.iloc[tune_size:x] 
        y_train = dl.disc_shifted_label(timeseries).iloc[tune_size:x]
    else: #sliding window with window size given in config.py
        X_train = X.iloc[x-(config.window_size):x]
        y_train = dl.disc_shifted_label(timeseries).iloc[x-(config.window_size):x]  
    X_test = X.iloc[x+1]
    y_test = dl.disc_shifted_label(timeseries).iloc[x+1]
    
    #Making code more robust aaginst nan values:
    X_train = np.nan_to_num(X_train) 
    y_train = np.nan_to_num(y_train) 
    X_test = np.nan_to_num(X_test) 
    y_test = np.nan_to_num(y_test)
    
    return X_train, X_test, y_train, y_test

def data_split_CV(X_train):
# Fixing the size of training and testsets manually is too expensive computationalwise
#    num_in_test = 5
#    test_size = float(num_in_test) / len(X_train)
#    n_splits = int((1//test_size)-1)
#    X_train = X_train.reset_index(drop=True)
# Therefore we set a fixed number of splits:
    n_splits = 5
    my_cv = TimeSeriesSplit(n_splits) #https://scikit-learn.org/stable/glossary.html#term-cv-splitter --> cross-validation generator¶
    return my_cv

