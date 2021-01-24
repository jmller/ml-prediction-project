#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test/Train Split
"""
import numpy as np
import pandas as pd

import discrete_layer as dl
from sklearn.model_selection import KFold
from sklearn.model_selection import PredefinedSplit

import config

def data_split_0(X, y, timeseries, config_list):
    tune_size = np.max(X[:].isna().sum())
    y = y.replace(np.nan, 0) #necessary; otherwise the hyperparameteroptimization will not work when randomwalk is given!
    X = X.replace(np.nan, 0) #idem    
    X_train = X.iloc[tune_size::2, :] #Take each second day from tune_size on
    y_train = dl.disc_shifted_label(timeseries).iloc[tune_size::2]    
    X_test = X.iloc[tune_size+1::2, :] #Take each second day from tune_size+1 on
    y_test = dl.disc_shifted_label(timeseries).iloc[tune_size+1::2]
    X_params_select = X.iloc[tune_size::5, :] #Take 20% for params_optimization
    y_params_select = y.iloc[tune_size::5]   
#    X_test= X_train
#    y_test= y_train
#    X_params_select = X_train
#    y_params_select = y_train
    return X_train, X_test, y_train, y_test, X_params_select, y_params_select

def data_split_1(X, y, timeseries, config_list):
    tune_size = np.max(X[:].isna().sum())
    y = y.replace(np.nan, 0)
    X = X.replace(np.nan, 0)    
    X_ten_fold = np.array_split(X.iloc[tune_size:, :], 100) #Take every second split
    y_ten_fold = np.array_split(dl.disc_shifted_label(timeseries).iloc[tune_size:], 10)
    X_train = pd.concat(X_ten_fold[0::2])
    y_train = pd.concat(y_ten_fold[0::2])
    X_test = pd.concat(X_ten_fold[1::2]) 
    y_test = pd.concat(y_ten_fold[1::2])
    X_params_select = pd.concat(X_ten_fold[0::5]) #Take 20% for params_optimization
    y_params_select = pd.concat(y_ten_fold[0::5])
#    X_test= X_train
#    y_test= y_train
#    X_params_select = X_train
#    y_params_select = y_train
    return X_train, X_test, y_train, y_test, X_params_select, y_params_select

def data_split_2(X, y, timeseries, config_list):
    tune_size = np.max(X[:].isna().sum())
    y = y.replace(np.nan, 0)
    X = X.replace(np.nan, 0)
    X_ten_fold = np.array_split(X.iloc[tune_size:, :], 10) #Take every second split
    y_ten_fold = np.array_split(dl.disc_shifted_label(timeseries).iloc[tune_size:], 10)
    X_train = pd.concat(X_ten_fold[0::2])
    y_train = pd.concat(y_ten_fold[0::2])
    X_test = pd.concat(X_ten_fold[1::2]) 
    y_test = pd.concat(y_ten_fold[1::2])
    X_params_select = pd.concat(X_ten_fold[0::5]) #Take 20% for params_optimization
    y_params_select = pd.concat(y_ten_fold[0::5])
#    X_test= X_train
#    y_test= y_train
#    X_params_select = X_train
#    y_params_select = y_train
    return X_train, X_test, y_train, y_test, X_params_select, y_params_select

def data_split_CV(X_train, X_params_select):
    pass
    return my_cv

