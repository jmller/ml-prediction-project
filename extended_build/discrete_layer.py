#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discrete Layer;
Continuous values will be discretized here
"""
import numpy as np

import continuous_layer as cl

def disc_layer(time_series):
    proc_disc = cl.cont_layer(time_series) #passes the information to the continuous layer
    #Implementing RSI Conditions 
    proc_disc['RSI'] = np.where(proc_disc['RSI'] < 30, +1, proc_disc['RSI'])
    proc_disc['RSI'] = np.where(proc_disc['RSI'] > 70, -1, proc_disc['RSI'])
    proc_disc['RSI'] = np.where(np.abs(proc_disc['RSI']) != 1, cl.cont_layer(time_series)['RSI'] - cl.cont_layer(time_series)['RSI'].shift(1), proc_disc['RSI'])
    #Implementing CCI Conditions
    proc_disc['CCI'] = np.where(proc_disc['CCI'] > 200, -1, proc_disc['CCI'])
    proc_disc['CCI'] = np.where(proc_disc['CCI'] < -200, +1, proc_disc['CCI'])
    proc_disc['CCI'] = np.where(np.abs(proc_disc['CCI']) != 1, cl.cont_layer(time_series)['CCI'] - cl.cont_layer(time_series)['CCI'].shift(1), proc_disc['CCI'])
    return np.sign(proc_disc) #discretize all remaining continuous values

def disc_label(timeseries):
    return np.sign(cl.cont_label(timeseries)) #discretize all labels

def disc_shifted_label(timeseries):
    return disc_label(timeseries).shift(-1)