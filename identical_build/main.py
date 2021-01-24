#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main-program containing the outer loop
Necessary libraries and modules are automatically imported
If an incorrect file is given, the program will automatically load the Nifty50.csv file
Most parameters can be set in config.py
"""
import pandas as pd
import numpy as np

import discrete_layer as dl
import data_split as ds
import random_forest_algo as ml_rf

import naive_bayes_algo as ml_nb
import svc_algo as ml_svc
import ann_algo as ml_ann

import config
import documentation

import openpyxl as op
from openpyxl import Workbook
from openpyxl.worksheet.table import Table, TableStyleInfo


import time


###
#Loading Data
###
try: 
    dataset_input = input("Please enter the filename: \n") #Nifty50.csv will be selected if no other input occurs
    df = pd.read_csv(dataset_input)
        
except:
    print("File not found!")
    print("Loading Nifty50.csv")
    dataset_input = "Nifty50.csv" 
    df = pd.read_csv(dataset_input)
    
###
# Data Cleaning
###
try:
    df = df.dropna()
    df = df.drop_duplicates(subset=['Open', 'High', 'Low', 'Close'], keep='first', inplace=False)    
    df = df[df.Volume != 0]
except:
    pass
    
# Main-loop calls the discrete layer and passes the loaded information;
# Call yields prepared X and y vectors
X=dl.disc_layer(df)
y=dl.disc_shifted_label(df)
tune_size = np.max(X[:].isna().sum())
timeseries = df
start_time = time.time() #Capturing runtime

###
# Mainloop: Calling the Functions
###

def all_params_loop(final_table, pos_neg_total):
    count = 0
    start_iteration = 0
    for y in range(start_iteration, np.shape(config.param_combinations)[0]):
        config_list = config.config_list_var(y)
        online_measures, series_frame, pos_neg_total = main_loop(config_list, final_table, count)
        if count == 0:
            final_table = final_table.append(series_frame)
            pos_neg_total = pos_neg_total.reset_index()
            print(pos_neg_total)
            pos_neg_total.to_excel('pos_neg_total.xlsx', index=False)
        else:        
            final_table.insert(count, count, series_frame)
            df = pd.read_excel('pos_neg_total.xlsx')   
            df = df.append(pos_neg_total)
            df.to_excel('pos_neg_total.xlsx', index=False)
        final_table.to_excel('%d_final_table.xlsx' %count)
        count+=1
    return print("All %s Parameters were tested.", count)
   
def main_loop(config_list, final_table, count): 
    print(" chosen_method: %s \n expanding_windows: %d \n optimize_method: %d \n randomizedsearch: %d \n" %(config_list['chosen_method'], config_list['expanding_window_ml'], config_list['optimize_method'], config_list['randomized_search']))
    online_measures = pd.DataFrame({'day': [], 'accuracy': [], 'f-measure': []})
    opt_modulo_params_list = []
    
    correct_pred = []
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    y_pred_list = []
    y_prob_list = []
    
    tpr_list = []
    fpr_list = []  
    
    opt_modulo_params = 0
    
    ###
    # Test/Train Split in every iteration x
    ###
    if config_list['chosen_data_split'] == 0:
        X_train, X_test, y_train, y_test, X_params_select, y_params_select = ds.data_split_0(X, y, timeseries, config_list)
    elif config_list['chosen_data_split'] == 1:
        X_train, X_test, y_train, y_test, X_params_select, y_params_select = ds.data_split_1(X, y, timeseries, config_list)  
    elif config_list['chosen_data_split'] == 2:
        X_train, X_test, y_train, y_test, X_params_select, y_params_select = ds.data_split_2(X, y, timeseries, config_list)          
    ###
    # Actual Training and Prediction
    ###
    if config_list['chosen_method'] == config.method [0]:
        y_pred, opt_modulo_params = ml_rf.classifier(X_train, y_train, X_test, X_params_select, y_params_select, tune_size, config_list, opt_modulo_params)
    elif config_list['chosen_method'] == config.method [1]:
        y_pred, opt_modulo_params = ml_svc.classifier(X_train, y_train, X_test, X_params_select, y_params_select, tune_size, config_list, opt_modulo_params)
    elif config_list['chosen_method'] == config.method [2]:
        y_pred, opt_modulo_params = ml_nb.classifier(X_train, y_train, X_test, X_params_select, y_params_select, tune_size, config_list, opt_modulo_params)
    elif config_list['chosen_method'] == config.method [3]:
        y_pred, opt_modulo_params = ml_ann.classifier(X_train, y_train, X_test, X_params_select, y_params_select, tune_size, config_list, opt_modulo_params)
    else:
        print("Select a classifier in the config file!")
   
    ###
    # Evaluation; Initially a function was written; Though the function was difficult to implement in the running program
    ###        
  
    try:
        y_pred = int(test_predictions)
        y_test = int(y_test) 
    except:
        pass
    
    y_pred_list.append(y_pred)
    
    #The following had to be corrected in the identical replication due to y_pred being a vector
    y_true_table = y_pred == y_test
    y_pred_plus = y_pred ==+1
    y_pred_minus = y_pred ==-1
    tp_list = list(y_true_table & y_pred_plus)
    tn_list = list(y_true_table & y_pred_minus)
    fp_list = list(~y_true_table & y_pred_plus)
    fn_list = list(~y_true_table & y_pred_minus)
    
    eval_tuple = (tp_list, fp_list, tn_list, fn_list)
    
    pos_prec = 0
    neg_prec = 0
    pos_recall = 0
    neg_recall = 0
    accuracy = 0
    f_measures = 0
    
    tp = tp_list.count(True)
    fp = fp_list.count(True)
    tn = tn_list.count(True)
    fn = fn_list.count(True)   
    
    print(tp)
    print(fp)
    print(tn)
    print(fn)

    try:
        #Accuracy
        accuracy = (tp + tn)/(tp + fp + tn +fn) 
        
        #pos Precision and Recall
        pos_prec = tp/(tp + fp)
        pos_recall = tp/(tp + fn)
         
        #F-Measures
        f_measures = (2*pos_prec*pos_recall)/(pos_prec + pos_recall)           
        
        #negative Precision andRecall
        neg_prec = tn/(tn + fn)
        neg_recall = tn/(tn + fp)

    except:
        print("Division by zero")   
    current_measures = pd.DataFrame({'accuracy': [accuracy], 'f-measure': [f_measures]})
    print(current_measures)
    online_measures = current_measures
     
        
    ###
    # Documentation
    ###
    if config.documentation == 1:        
        series_frame = documentation.concat_results(online_measures, config_list, dataset_input, start_time)
        pos_neg_total= pd.DataFrame({'tp':[tp], 'fp':[fp], 'tn':[tn], 'fn':fn})
        print(pos_neg_total)
        
        if config_list['optimize_method'] == 1:
            opt_modulo_params_list = pd.Series(opt_modulo_params)            
            opt_modulo_params_list.to_excel('%d_opt_modulo_params_list.xlsx' %count)        
        online_measures.iloc[-1]
    print("Mainloop was executed")
    return online_measures, series_frame, pos_neg_total
    
final_table = pd.DataFrame([])
pos_neg_total = pd.DataFrame([])
if config.all_params_loop == 1:    
    all_params_loop(final_table, pos_neg_total)
else:    
    config_list = config.config_list()
    online_measures, series_frame, pos_neg_total = main_loop(config_list, final_table, count = 0)
    final_table = final_table.append(series_frame)
    final_table.to_excel('final_table.xlsx')
    pos_neg_total.to_excel('pos_neg_total.xlsx')

print("Runtime: --- %s seconds ---" % (time.time() - start_time))
