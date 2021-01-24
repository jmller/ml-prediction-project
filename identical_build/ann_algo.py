#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import itertools

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense, LSTM

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import data_split


###
#Create Feedforward Network
###

#https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
def create_model(learn_rate = 0.001, momentum = 0, hidden_layer_n = 100, ):
    model = Sequential()
    model.add(InputLayer(input_shape = 10))
    model.add(Dense(hidden_layer_n, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = SGD(lr = learn_rate, momentum = momentum)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model


def classifier(X_train, y_train, X_test, X_params_select, y_params_select, tune_size, config_list, opt_modulo_params):
    ###
    # Hyperparameterarrays
    ###    
 
    learn_rate = [0.1]
    momentum = [float(x) for x in np.linspace(start = 0, stop = 0.9, num = 4)]
    hidden_layer_n = [int(x) for x in np.linspace(start = 10, stop = 100, num = 4)]
    
    param_grid = {'learn_rate': learn_rate,
                  'momentum': momentum,
                  'hidden_layer_n': hidden_layer_n,
                  }

    ###
    # Training and Fitting
    ###    
    early_stopping = EarlyStopping(monitor='accuracy', patience= 500)
    model = KerasClassifier(build_fn = create_model, epochs = 10000)
    if config_list['optimize_method'] == 1:
        if config_list['randomized_search'] == 1:    
            grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv = 2, n_iter = 25)
        else:
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = 2)
        
        model = grid.fit(X_params_select, y_params_select, epochs = 10000, callbacks=[early_stopping], verbose = 0)
        opt_modulo_params = grid.best_params_
        #model = create_model(**opt_modulo_params)
        print(opt_modulo_params)
    
    else:
        model = create_model()
        model.fit(X_train, y_train, epochs = 10000, callbacks=[early_stopping], verbose = 0)        
    
    predictions_prob = model.predict(X_test) #bugfix: https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras
    print(predictions_prob)
    y_pred = np.where(predictions_prob>=0.5, 1, -1)
    y_pred = list(itertools.chain(*y_pred))
    y_pred = np.array(y_pred)
    print(y_pred)
    print(type(y_pred))
    return y_pred, opt_modulo_params
