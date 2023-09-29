#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:08:14 2023

@author: dimitrios
"""


import pandas as pd
import numpy as np
import requests

class ForexData:
    def __init__(self, ticker_symbols, count=5000, granularity='H12', timezone='Europe/London'):
        self.ticker_symbols = ticker_symbols
        self.access_token ='37c8a10610c3db7afb9b3153a83b2998-a3bec539452b400b95a0133422f322ac'
        self.count = count
        self.granularity = granularity
        self.timezone = timezone
        self.data = None

    def get_historical_data(self):
        url = f'https://api-fxpractice.oanda.com/v3/instruments/{self.ticker_symbols}/candles'
        # https://api-fxtrade.oanda.com
        headers = {'Authorization': f'Bearer {self.access_token}'}
        params = {'count': self.count, 'price': 'M', 'granularity': self.granularity, 'timezone': self.timezone}
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        # print(data)
        return data
    
    def get_current_data(self):
        url = f'https://api-fxpractice.oanda.com/v3/instruments/{self.ticker_symbols}/candles/latest'
        headers = {'Authorization': f'Bearer {self.access_token}'}
        params = {'price': 'M', 'granularity': self.granularity, 'timezone': self.timezone}
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        return data

    def retrieve_data(self):
        try:
            data = self.get_historical_data()
            candles = data['candles']
            prices = []
            for candle in candles:
                time = pd.to_datetime(candle['time'])
                volume = candle['volume']
                high = candle['mid']['h']
                low = candle['mid']['l']
                close_price = candle['mid']['c']
                open_price = candle['mid']['o']
                prices.append([time, volume, high, low, close_price, open_price])
    
            df = pd.DataFrame(prices, columns=['Time', 'Volume', 'High', 'Low', 'Close', 'Open'])
            df.set_index('Time', inplace=True)
            self.data = df
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            self.data = None


    def get_exchange_rate(self):
        return self.data['Close']

