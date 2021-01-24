#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

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
from tensorflow.keras.callbacks import EarlyStopping

import data_split


###
#Create Feedforward Network
###

#https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
def create_model(learn_rate = 0.01, momentum = 0, hidden_layer_n = 100):
    model = Sequential()
    model.add(InputLayer(input_shape = 10))
    model.add(Dense(hidden_layer_n, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = SGD(lr = learn_rate, momentum = momentum)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model


def classifier(X_train, y_train, X_test, x, tune_size, config_list, opt_modulo_params):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    ###
    # Hyperparameterarrays
    ###    
 
    learn_rate = [0.1]
    momentum = [float(x) for x in np.linspace(start = 0, stop = 0.9, num = 3)]
    hidden_layer_n = [int(x) for x in np.linspace(start = 10, stop = 100, num = 3)]
    
    param_grid = {'learn_rate': learn_rate,
                  'momentum': momentum,
                  'hidden_layer_n': hidden_layer_n,
                  }

    ###
    # Training and Fitting
    ###    
    early_stopping = EarlyStopping(monitor='accuracy', patience = 200)
    model = KerasClassifier(build_fn = create_model, epochs = 5000)
    if x > tune_size+config.min_samples_opt and x%200 == 0 and config_list['optimize_method'] == 1: #Hyperparameters will be optimized every 10 Ticks
 
        if config_list['randomized_search'] == 1:    
            grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv = data_split.data_split_CV(X_train), n_iter = 25)
        else:
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = data_split.data_split_CV(X_train))
        
        model = grid.fit(X_train, y_train, epochs = 5000, callbacks=[early_stopping], verbose = 0)
        opt_modulo_params = grid.best_params_
        predictions_prob = model.predict(np.array([X_test])) #bugfix: https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras
    
    else:
        if opt_modulo_params == 0:
            model = create_model()
        else:
            model = create_model(**opt_modulo_params)
        model.fit(X_train, y_train, epochs = 5000, callbacks=[early_stopping], verbose = 0)        
        predictions_prob = model.predict_step(np.array([X_test])) #bugfix: https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras
    y_pred = np.where(predictions_prob>=0.5, 1, -1)
    return y_pred, opt_modulo_params
