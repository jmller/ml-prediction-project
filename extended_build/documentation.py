#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Documentation
"""
###
#Used Libraries
###
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import TextBox
from matplotlib.offsetbox import AnchoredText

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import tech_indicators as ti
import continuous_layer as cl
import discrete_layer as dl
import random_forest_algo as ml_rf
import naive_bayes_algo as ml_nb

import config

import time

def save_doc(config_list, online_measures, dataset_input, df, count, opt_modulo_params_list):#executed_algo, expanding_window_ml, optimize_method, randomizedsearch, chosen_method, dataset_input, df, count, documentation = config.documentation):
    if config.documentation == 1:
        dpi_scale = 3
        fig = plt.figure(figsize=(8, 6), dpi=100*dpi_scale)
        
        #for a better comparison, the y-axis is set to the boundaries of [0, 1]   
        plt.ylim(0, 1) 
        plt.plot(online_measures['day'], online_measures['accuracy'],'r', label = 'accuracy')
        plt.plot(online_measures['day'], online_measures['f-measure'],'b', label = 'f-measure')
        plt.title(config_list['chosen_method'])
        plt.ylabel('accuracy')
        plt.xlabel('samples')
        plt.legend()

        insert = 'expanding_window = {}'.format(config_list['expanding_window_ml'])+ "\noptimize_method = {}".format(config_list['optimize_method']) + '\nrandomized_search = {}'.format(config_list['randomized_search']) + '\ndataset = {}'.format(dataset_input) + '\nsamples = {}'.format(len(df))
        plt.text(0.65, 0.2, insert, ha='left', bbox=dict(facecolor='grey', alpha=0.5), transform=fig.transFigure, wrap=True)
        plt.savefig('%d_eval_results.png' %count, dpi=100*dpi_scale)
        online_measures.to_excel('%d_eval_results.xlsx' %count)      
            
def concat_results(online_measures, config_list, dataset_input, start_time):
    #Calculate the mean over the accuracy and F-Measure
    mean_measures = np.mean(online_measures.drop(columns = ['day']))
    df_config_list = pd.DataFrame(config_list)
    df_config_list = df_config_list.drop(columns = ['method','all_params_loop', 'documentation', 'window_size']).iloc[0]
    one_row_final_table = df_config_list.append(mean_measures)
    one_row_final_table = one_row_final_table.append(pd.Series({'dataset': dataset_input}))
    one_row_final_table = one_row_final_table.append(pd.Series({'runtime in sec': (time.time() - start_time)}))
    
    series_frame = one_row_final_table.to_frame()

    return series_frame

def AUC():
    pass        
        