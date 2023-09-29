#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:53:02 2023

@author: dimitrios
"""


import yfinance as yf
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from textblob import TextBlob
# from ta import CandlePatterns
from math import isclose
import random
# from fibonacci import fib_retracement

from scipy.signal import savgol_filter

import pandas as pd
import numpy as np
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from datetime import datetime, timedelta

import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from oanda_api import OandaAPI
import json
import requests
import pandas as pd
import numpy as np
import oandapyV20
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from features import calculate_derivatives
from forex_data import ForexData
from forex_model import ForexModel

# Example usage:
api_token = 'your-token'

account_id = 'your-account_id'

oanda_api = OandaAPI(api_token, account_id)






# Usage example

ticker_exotic = [
    'EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'USDCHF=X', 'USDCAD=X', 'AUDUSD=X', 'NZDUSD=X',
    'EURGBP=X', 'EURJPY=X', 'EURCHF=X', 'EURAUD=X', 'EURCAD=X', 'EURNZD=X',
    'GBPJPY=X', 'GBPCHF=X', 'GBPAUD=X', 'GBPCAD=X', 'GBPNZD=X',
    'AUDJPY=X', 'AUDCHF=X', 'AUDCAD=X', 'AUDNZD=X',
    'CADJPY=X', 'CADCHF=X', 'CADAUD=X', 'CADNZD=X',
    'NZDJPY=X', 'NZDCHF=X', 'NZDAUD=X', 'NZDCAD=X',
    'CHFJPY=X', 'CHFAUD=X', 'CHFCAD=X', 'CHFNZD=X',
    'USDTRY=X', 'USDMXN=X', 'USDBRL=X', 'USDZAR=X',
    'USDRUB=X', 'USDINR=X', 'USDCNY=X', 'USDTHB=X'
]




ticker_symbols = [
     # 'BTC-USD',
    'EURUSD=X',  'GBPUSD=X',
    'USDCHF=X', 'USDCAD=X', 'AUDUSD=X',
       'EURGBP=X', 'EURCHF=X', 'EURCAD=X',
    'GBPCHF=X', 'GBPCAD=X', 
      'AUDNZD=X'  # 'USDJPY=X', 'USDTRY=X', 'USDMXN=X', 'USDBRL=X', 'USDZAR=X',
        # 'USDRUB=X', 'USDINR=X', 'USDCNY=X', 'USDTHB=X'
]

ticker_mapping = {
    'EURUSD=X': 'EUR_USD',
    'USDJPY=X': 'USD_JPY',
    'GBPUSD=X': 'GBP_USD',
    'USDCHF=X': 'USD_CHF',
    'USDCAD=X': 'USD_CAD',
    'AUDUSD=X': 'AUD_USD',
    'EURGBP=X': 'EUR_GBP',
    'EURCHF=X': 'EUR_CHF',
    'EURCAD=X': 'EUR_CAD',
    'GBPCHF=X': 'GBP_CHF',
    'GBPCAD=X': 'GBP_CAD',
    'AUDJPY=X': 'AUD_JPY',
    'AUDNZD=X': 'AUD_NZD',
}



# ticker_symbols = 'EUR_USD'  # Replace with your desired ticker symbols
# count = 5000  # Replace with the desired count value
# granularity = 'H1'  # Replace with the desired granularity
# timezone = 'Europe/London'  # Replace with your desired time zone


# forex_data = ForexData(ticker_symbols, count=count, granularity=granularity, timezone=timezone)

# # Now you can use the forex_data instance to retrieve and work with the data
# forex_data.retrieve_data()
# exchange_rate = forex_data.get_exchange_rate()
# derivatives = forex_data.calculate_derivatives()




import warnings

def calculate_pips(pair, price):
    if pair.endswith('JPY'):
        # For JPY pairs (USDJPY, EURJPY, etc.)
        # Pips are calculated with two decimal places
        return round((price - 1) * 100, 2)
    else:
        # For other currency pairs (EURUSD, GBPUSD, etc.)
        # Pips are calculated with four decimal places
        return round((price - 1) * 10000, 4)
    

# ticker_symbols = 'EUR_USD'  # Replace with your desired ticker symbols
count = 5000  # Replace with the desired count value
granularity = 'D'  # Replace with the desired granularity
timezone = 'Europe/London'  # Replace with your desired time zone
warnings.filterwarnings("ignore", message="Converting data to scipy sparse matrix.")
from datetime import time
cou=0
cou2=0
cou3=0
cou4=0
previous_timestamps = {}
action={}
sum_of_orders_vec=[]
profit_loss=0
flag={}
enoum={}
gbmt={}

for ticker1 in ticker_symbols:
    ticker=ticker_mapping[ticker1]
    flag[ticker]=[]
    flag[ticker]=0
    enoum[ticker]=[]
    enoum[ticker]=0
    gbmt[ticker]=[]



while True:
        for ticker1 in ticker_symbols:
            pred_values_array = []
            current_prices_array = []
            high_prices_array = []
            low_prices_array = []
            timestamps_array = []
            pred_diff = []
            actual_diff = []
            sum_of_orders_=0
            sum_stratege2=0
            ticker=ticker_mapping[ticker1]
            try:
                forex_data = ForexData(ticker, count=count, granularity=granularity, timezone=timezone)
                forex_data.retrieve_data()
                exchange_rate = forex_data.get_exchange_rate()
                if exchange_rate.empty:
                    raise Exception("Empty exchange rate data")
            except Exception as e:
                print(f"Error occurred for {ticker}: {str(e)}")
                continue  # Continue to the next ticker symbol
    
            # input_data = forex_data.calculate_derivatives().iloc[-1]
            # timestamp = input_data.name.strftime("%Y-%m-%d %H:%M:%S")
    
    
            train_data = forex_data.data
            forex_derivatives=calculate_derivatives(train_data)
            input_data =forex_derivatives.iloc[-1]
            current_price=input_data['X']
            timestamp = input_data.name.strftime("%Y-%m-%d %H:%M:%S")
            if enoum[ticker]==0:
                forex_model = ForexModel(forex_derivatives)
                forex_model.prepare_data()
                print('===============================')
                forex_model.train_model()
                gbmt[ticker]=forex_model.gbm
                print('===============================')
                enoum[ticker]=1
            
            if ticker in previous_timestamps and timestamp == previous_timestamps[ticker]:
                flag[ticker]=1
                print('Profit/Loss:', profit_loss)
                continue  # Skip the current iteration if the timestamp hasn't changed for this ticker
            if flag[ticker]==1 : 
                try:
                    if ticker in action and action[ticker]!=[]:
                        cancellation_response = oanda_api.close_trade(action[ticker][0][2])
                    print('Order closed:', cancellation_response)
                    if cancellation_response!=None:
                        profit_loss += float(cancellation_response.get('orderFillTransaction', {}).get('pl'))
                    print('Profit/Loss:', profit_loss)
                except V20Error as e:
                    print('Error closing order:', e)
                action[ticker] = []
                flag[ticker]=0
            
            
            forex_model = ForexModel(forex_derivatives)
            forex_model.prepare_data()
            # if enoum[ticker]==0:
            #     print('===============================')
            #     forex_model.train_model()
            #     gbmt[ticker]=forex_model.gbm
            #     print('===============================')
            #     enoum[ticker]==1
            
            pred_value=gbmt[ticker].predict(input_data.drop('X'),gbmt[ticker].best_iteration)#num_iteration=10000)#gbm.best_iteration)
            enoum[ticker]=0
            pred_value_in=np.exp(pred_value[0])
            pip_difference = abs(current_price - pred_value_in)
            threshold = 0.2 if ticker.endswith('JPY') else 0.001
            # print(f'{ticker} pip difference {current_price - pred_value_in}')
            if pip_difference >= threshold: #and ( input_data.name.time()<time(12, 0) and input_data.name.time()>time(17, 0) ):
                    # order_type = "BUY" if input_data['X'][0] < pred_value[0] else "SELL"
                print('===============================')
                # print(f"Place a {order_type} order for {ticker} - Current Price: {input_data['X'][0]:.6f}, Predicted Price: {pred_value[0]:.6f}, Time: {timestamp}")
                order_type = "BUY" if current_price < pred_value_in else "SELL"
                print(f"Place a {order_type} order for {ticker} - Current Price: {current_price:.6f}, Predicted Price: {pred_value_in:.6f}, Time: {timestamp}")
            
            
            
            
                sign= 1 if order_type=="BUY" else -1
                #if ticker not in action:
                if flag[ticker]==0 :
                    if ticker not in action:
                        action[ticker]=[]
                    try:
                        take_profit=pred_value_in-sign*np.abs(current_price-pred_value_in)*10/100
                        order_response = oanda_api.place_order(order_type='MARKET', instrument=ticker,  units=sign*1000, stop_loss=None, take_profit=take_profit)
                        trade_id = order_response.get('orderFillTransaction', {}).get('tradeOpened', {}).get('tradeID')
                        print('Order placed:', order_response)
                        print('Order ID:', trade_id)  # Save the order ID
                    except V20Error as e:
                        print('Error placing order:', e)
                    action[ticker].append([order_type,timestamp,trade_id])
                    
    
            
                
            # if ticker not in action:
            #     # action[ticker] = []
            #     # Place an order
            #     try:
            #         order_response = oanda_api.place_order(order_type, ticker_mapping[ticker], units)
            #         print('Order placed:', order_response)
            #     except V20Error as e:
            #         print('Error placing order:', e)
                
            #     action[ticker].append([order_type, timestamp])
    
            # elif action[ticker][-1][0] != order_type:
            #     action[ticker].append([order_type, timestamp])
                # Close the existing order
                # if order_id:
                    # try:
                    #     cancellation_response = oanda_api.close_order(order_id)
                    #     print('Order closed:', cancellation_response)
                    # except V20Error as e:
                    #     print('Error closing order:', e)
        
            # threshold2= threshold if  order_type == "BUY" else -threshold
        
            
      
            # bet=(current_price+predictions[0])/2
                # print(bet)
                # if data2['High'].values[0]<bet and bet>data2['Low'].values[0] :
                #     print("YES")
                #     count+=1
                # else:
                #     print("NO")
                #     count2+=1
                # print(count,count2)
                # print('===============================')
                
        
            previous_timestamps[ticker] = timestamp




column_sums = [print(sum(x)) for x in zip(*sum_of_orders_vec)]
