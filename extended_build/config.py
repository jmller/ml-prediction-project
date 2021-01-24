#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential



import itertools

###
#Global Variables
###

all_params_loop = 1
documentation = 1
min_samples_opt = 20 # lead time (in days) before optimization starts

###
# User set parameters
###

window_size = 200 
method = [BernoulliNB, RandomForestClassifier, SVC, Sequential]
chosen_method = method[2]
expanding_window_ml = 0
optimize_method = 0
randomized_search = 1


###
# All parameters generation
###

option_param_method = list(range(0, np.size(method))) #number of used ML-methods
option_param_on_off = [1, 0]
param_combinations = list(itertools.product(option_param_method, option_param_on_off, option_param_on_off))#, option_param_on_off))

def config_list_var(x):

    chosen_method = method[param_combinations[x][0]]
    expanding_window_ml = param_combinations[x][1]
    optimize_method = param_combinations[x][2]
    randomized_search = 0 #param_combinations[x][3]
    
    config_list = {'all_params_loop': all_params_loop,
               'window_size': window_size,
               'method': method,
               'chosen_method': chosen_method,
               'expanding_window_ml': expanding_window_ml,
               'optimize_method': optimize_method,
               'randomized_search': randomized_search,
               'documentation': documentation}
    return config_list

def config_list():
    config_list = {'all_params_loop': all_params_loop,
                   'window_size': window_size,
                   'method': method,
                   'chosen_method': chosen_method,
                   'expanding_window_ml': expanding_window_ml,
                   'optimize_method': optimize_method,
                   'randomized_search': randomized_search,
                   'documentation': documentation}
    return config_list