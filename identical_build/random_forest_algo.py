"""
Created on Fri Jul 10 11:40:44 2020
@author: johannes
"""
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import config
import data_split


def classifier(X_train, y_train, X_test, X_params_select, y_params_select, tune_size, config_list, opt_modulo_params):

    ###
    # Hyperparametergrid
    ###
    
    #https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    # Number of trees in random forest 
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 6)]
    # Number of features to consider at every split
    # 3 according to the paper
    max_features = [3]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num = 4)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 8]
    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               #'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
               }
  
    #Learning and Tuning
    classifier = RandomForestClassifier(n_jobs = -1)
    if config_list['optimize_method'] == 1:
        if config_list['randomized_search'] == 1:
            grid = RandomizedSearchCV(estimator= classifier, param_distributions=random_grid, cv = 2, n_iter=25, n_jobs=-1)
        else:
            grid = GridSearchCV(estimator= classifier, param_grid=random_grid, cv = 2, n_jobs=-1)
        
        classifier = grid.fit(X_params_select, y_params_select)  
        opt_modulo_params = grid.best_params_
        classifier = RandomForestClassifier(**opt_modulo_params, n_jobs = -1)
        print(opt_modulo_params)
        
    else:
        classifier = RandomForestClassifier(n_estimators = 100, max_depth = 50, n_jobs = -1,) # arbitrarily set parameters; limits growth of trees --> Avoids overfitting 
    classifier.fit(X_train, y_train)    
    y_pred = classifier.predict(X_test) #bugfix: https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras
    return y_pred, opt_modulo_params
