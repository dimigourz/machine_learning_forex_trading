#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:09:50 2023

@author: dimitrios
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

class ForexModel:
    def __init__(self, data):
        self.data = data
        self.df = None
        self.X = None
        self.y = None
        self.lgb_train = None
        self.lgb_val = None
        # self.params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'regression',
        #     'metric': {'l2'},
        #     'num_leaves': 4,
        #     'learning_rate': 0.1,
        #     # 'feature_fraction': 0.9,
        #     # 'bagging_fraction': 0.8,
        #     # 'bagging_freq': 5,
        #     # 'verbose': -1,
        #     # 'predict_disable_shape_check': 'true'
        #     # 'silent': 0  # Set silent to True to disable all output
        # }
        callbacks = [lgb.callback.log_evaluation(period=100)]  # Log evaluation results every 100 iterations
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l1'},
            'num_leaves': 32,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 1,
            'verbose': -1,
            'callbacks': callbacks  # Use the 'log_evaluation()' callback
        }
                
        self.gbm = None

    def prepare_data(self):
        self.df = self.data
        # print( self.df)
        # # Get the last row of the DataFrame
        # last_entry = self.df.iloc[[-1]]
        
        # # Shuffle the remaining rows
        # shuffled_df = self.df.iloc[:-1].sample(frac=1, random_state=random.seed())
        
        # # Append the last entry to the shuffled DataFrame
        # shuffled_df = shuffled_df.append(last_entry)
        # self.df=shuffled_df
        # print( self.df)
        
        self.df['y'] = np.log(self.df['X'].shift(-1))
        self.df = self.df.iloc[:-1]
        self.X = self.df.drop(columns=['y', 'X']) 
        self.y = self.df['y']

        # Splitting the data into training and validation sets (80-20 split in this case)
        # X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        split_index =4998#int(0.8 * forex_data.data.shape[0])
        X_train = self.X#[:split_index]
        X_val = self.X[split_index:]
        y_train = self.y#[:split_index]
        y_val = self.y[split_index:]
        
        
        # Creating the datasets for LightGBM
        # print(X_val,y_val)
        
        self.lgb_train = lgb.Dataset(X_train, y_train)
        self.lgb_val = lgb.Dataset(X_val, y_val)

        # self.lgb_train = lgb.Dataset(self.X, self.y)
  
        
    def data_(self):
        return self.df

    def train_model(self):
        # def logloss_objective(y_pred, dtrain):
        #     y_true = dtrain.get_label()
        #     p = 1. / (1. + np.exp(-y_pred))
        #     grad = p - y_true
        #     hess = p * (1. - p)
        #     return grad, hess
        callbacks = [lgb.callback.log_evaluation(period=100)]  # Log evaluation results every 100 iterations
        
        self.gbm = lgb.train(self.params, 
                             self.lgb_train, 
                             num_boost_round=30000, 
                             valid_sets=[self.lgb_train, self.lgb_val],  
                             callbacks=callbacks)  # Use the 'log_evaluation()' callback


    def predict(self, input_data):
        # self.params['predict_disable_shape_check'] = True
        return self.gbm.predict(input_data.drop('X'), num_iteration=10000)#self.gbm.best_iteration)

