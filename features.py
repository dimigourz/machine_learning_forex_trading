#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:04:20 2023

@author: dimitrios
"""

import pandas as pd
import numpy as np
    

def calculate_derivatives(data):
    data.index = pd.to_datetime(data.index)  # Convert index to datetime format
    exchange_rate = pd.to_numeric(data['Close'], errors='coerce')
    # dt = (data.index[-1] - data.index[0]) / (len(data) - 1)
    dt = np.diff(data.index).mean()
    dt = dt.total_seconds()
    dX_dt = np.gradient(exchange_rate)
    d2X_dt2 = np.gradient(dX_dt, dt)
    d3X_dt3 = np.gradient(d2X_dt2, dt)
    d4X_dt4 = np.gradient(d3X_dt3, dt)
    d5X_dt5 = np.gradient(d4X_dt4, dt)
    d6X_dt6 = np.gradient(d5X_dt5, dt)
    d7X_dt7 = np.gradient(d6X_dt6, dt)
    d8X_dt8 = np.gradient(d7X_dt7, dt)
    d9X_dt9 = np.gradient(d8X_dt8, dt)
    d10X_dt10 = np.gradient(d9X_dt9, dt)
    d11X_dt11 = np.gradient(d10X_dt10, dt)
    # These may need to be adjusted depending on your data
    # window_length = 5
    # polyorder = 4
    
    # # Calculate derivatives
    # dX_dt = savgol_filter(exchange_rate, window_length, polyorder, deriv=1, delta=dt)
    # d2X_dt2 = savgol_filter(exchange_rate, window_length, polyorder, deriv=2, delta=dt)
    # d3X_dt3 = savgol_filter(exchange_rate, window_length, polyorder, deriv=3, delta=dt)
    # d4X_dt4 = savgol_filter(exchange_rate, window_length, polyorder, deriv=4, delta=dt)
    # d5X_dt5 = savgol_filter(exchange_rate, window_length, polyorder, deriv=5, delta=dt)
    # d6X_dt6 = savgol_filter(exchange_rate, window_length, polyorder, deriv=6, delta=dt)
    # d7X_dt7 = savgol_filter(exchange_rate, window_length, polyorder, deriv=7, delta=dt)
    # d8X_dt8 = savgol_filter(exchange_rate, window_length, polyorder, deriv=8, delta=dt)
    # d9X_dt9 = savgol_filter(exchange_rate, window_length, polyorder, deriv=9, delta=dt)
    # dX_dt = np.gradient(exchange_rate**(11) + exchange_rate**(10)+ exchange_rate**(9)+ exchange_rate**(8)
    #                     +exchange_rate**(7) + exchange_rate**(6)+ exchange_rate**(5)+ exchange_rate**(4)
    #                     +exchange_rate**(3) + exchange_rate**(2)+ exchange_rate**(1)+ exchange_rate**(0), dt)
    # d2X_dt2=11*exchange_rate**(10) +10* exchange_rate**(9)+ 9*exchange_rate**(8)+ 8*exchange_rate**(7)+7*exchange_rate**(6) + 6*exchange_rate**(5)+ 5*exchange_rate**(4)+ 4*exchange_rate**(3) +3*exchange_rate**(2) + 2*exchange_rate**(1)+ exchange_rate**(0)
    # d3X_dt3=10*11*exchange_rate**(9) +9*10* exchange_rate**(8)+ 9*8*exchange_rate**(7)+ 7*8*exchange_rate**(6)+7*6*exchange_rate**(5) + 6*5*exchange_rate**(4)+ 5*4*exchange_rate**(3)+ 4*3*exchange_rate**(2) +3*2*exchange_rate**(1) + 2*exchange_rate**(0)
    # d4X_dt4=10*11*9*exchange_rate**(8) +9*10*8* exchange_rate**(7)+ 9*8*7*exchange_rate**(6)+ 7*8*6*exchange_rate**(5)+7*6*5*exchange_rate**(4) + 6*5*4*exchange_rate**(3)+ 5*4*3*exchange_rate**(2)+ 4*3*2*exchange_rate**(1) +3*2*exchange_rate**(0)

    
    hexchange_rate=pd.to_numeric(data['High'], errors='coerce')
    hdX_dt = np.gradient(hexchange_rate, dt)
    hd2X_dt2 = np.gradient(hdX_dt, dt)
    hd3X_dt3 = np.gradient(hd2X_dt2, dt)
    lexchange_rate=pd.to_numeric(data['Low'], errors='coerce')
    ldX_dt = np.gradient(lexchange_rate, dt)
    ld2X_dt2 = np.gradient(ldX_dt, dt)
    ld3X_dt3 = np.gradient(ld2X_dt2, dt)
    
    mexchange_rate=(hexchange_rate-lexchange_rate)
    mdX_dt = np.gradient(mexchange_rate, dt)
    md2X_dt2 = np.gradient(mdX_dt, dt)
    md3X_dt3 = np.gradient(md2X_dt2, dt)
    
    oexchange_rate=pd.to_numeric(data['Open'], errors='coerce')
    odX_dt = np.gradient(oexchange_rate, dt)
    od2X_dt2 = np.gradient(odX_dt, dt)
    od3X_dt3 = np.gradient(od2X_dt2, dt)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    df = pd.DataFrame({
        "X":exchange_rate,
        
        "X1":np.log(exchange_rate),


        

        
        
        # "X2":exchange_rate+(2 * np.random.random(exchange_rate.shape) - 1)*0.01,
        # "X3":exchange_rate+(2 * np.random.random(exchange_rate.shape) - 1)*0.01,
        # "X4":exchange_rate+(2 * np.random.random(exchange_rate.shape) - 1)*0.01,
        # "X5":exchange_rate+(2 * np.random.random(exchange_rate.shape) - 1)*0.01,
        # "X6":exchange_rate+(2 * np.random.random(exchange_rate.shape) - 1)*0.01,
        # "X7":exchange_rate+(2 * np.random.random(exchange_rate.shape) - 1)*0.01,
        # "X8":exchange_rate+(2 * np.random.random(exchange_rate.shape) - 1)*0.01,
        # "X9":exchange_rate+(2 * np.random.random(exchange_rate.shape) - 1)*0.01,
        # "X10":exchange_rate+(2 * np.random.random(exchange_rate.shape) - 1)*0.01,

        
        # "X3":exchange_rate,

        # "X4":np.exp((2 * np.random.random(exchange_rate.shape) - 1)*2),

        # "X2": exchange_rate**(11) + exchange_rate**(10)+ exchange_rate**(9)+ exchange_rate**(8)
        #     +exchange_rate**(7) + exchange_rate**(6)+ exchange_rate**(5)+ exchange_rate**(4)
        #     +exchange_rate**(3) + exchange_rate**(2)+ exchange_rate**(1)+ exchange_rate**(0),
        # "d4X_dt4":(d4X_dt4)*100000000000000,
        # "d5X_dt5":(d5X_dt5)*100000000000000,
        # "d6X_dt6":(d6X_dt6)*100000000000000,
        # "d7X_dt7":(d7X_dt7)*100000000000000,
        # "d8X_dt8":(d8X_dt8)*100000000000000,
        # "d9X_dt9":(d9X_dt9)*100000000000000,
        # "d10X_dt10":(d10X_dt10)*100000000000000,
        # "d11X_dt11":(d11X_dt11)*100000000000000,

        # "tanhdt":np.tanh(exchange_rate)**2,
        # "htanhdt":np.tanh(hexchange_rate)**2,
        # "ltanhdt":np.tanh(lexchange_rate)**2,
        # "mtanhdt":np.tanh(mexchange_rate)**2,
        # # "otanhdt":np.tanh(self.data['Open'])**2,
        # "sigmoiddt":sigmoid(exchange_rate),
        # "hsigmoiddt":sigmoid(hexchange_rate)**2,
        # "lsigmoiddt":sigmoid(lexchange_rate)**2,
        # "mtsigmoiddt":sigmoid(mexchange_rate)**2,
        # # "osigmoiddt":sigmoid(self.data['Open'])**2,
        # "sigmoiddt":sigmoid(exchange_rate**2),
        # "hsigmoiddt":sigmoid(hexchange_rate**2),
        # "lsigmoiddt":sigmoid(lexchange_rate**2),
        # "mtsigmoiddt":sigmoid(mexchange_rate**2),

        # "minus_X": -exchange_rate,
        # "sqrt_X": exchange_rate**0.5,
        # "dX_dt":  (dX_dt),
        # "tanhdX_dt": np.tanh(dX_dt),
        # "sigmoiddX_dt": sigmoid(dX_dt),
        # "dX_6th":np.tanh(dX_dt**(-7) + dX_dt**(-6)+ dX_dt**(-5)+ dX_dt**(-4)),
        # "dX_7th": np.tanh(dX_dt**(-6)+ dX_dt**(-5)+ dX_dt**(-4)+ dX_dt**(-3)),
        # "dsigmoidX_6th":sigmoid(dX_dt**(-7) + dX_dt**(-6)+ dX_dt**(-5)+ dX_dt**(-4)),
        # "dsigmoidX_7th": sigmoid(dX_dt**(-6)+ dX_dt**(-5)+ dX_dt**(-4)+ dX_dt**(-3)),
        # "dX_6th-":np.tanh(dX_dt**(7) + dX_dt**(6)+ dX_dt**(5)+ dX_dt**(4)),
        # "dX_7th-": np.tanh(dX_dt**(6)+ dX_dt**(5)+ dX_dt**(4)+ dX_dt**(3)),
        # "dsigmoidX_6th-":sigmoid(dX_dt**(7) + dX_dt**(6)+ dX_dt**(5)+ dX_dt**(4)),
        # "dsigmoidX_7th-":sigmoid(dX_dt**(6)+ dX_dt**(5)+ dX_dt**(4)+ dX_dt**(3)),
        # "dlogX":np.log(dX_dt**(6)+ dX_dt**(5)+ dX_dt**(4)+ dX_dt**(3)),
        # "dsin_X": np.sin(dX_dt**(6)+ dX_dt**(5)+ dX_dt**(4)+ dX_dt**(3)),
        # "d2X_dt2":(d2X_dt2)*1000000000000,
        # "d3X_dt3":(d3X_dt3)*1000000000000,
        # "d3X_dt3+": d3X_dt3+dX_dt+d2X_dt2+exchange_rate*d2X_dt2,

        # "X_squared": sigmoid(exchange_rate**1+ exchange_rate**(4)+ exchange_rate**(3)+ exchange_rate**(2)),
        # "X_cubed": exchange_rate**3,
        # "X_4th": exchange_rate**4,
        # "X_5th": exchange_rate**5,
        # 0.0001
        # 0.029560000000000038
        # "X_6th":np.tanh(exchange_rate**(7) + exchange_rate**(6)+ exchange_rate**(5)+ exchange_rate**(4)),
        # "X_7th": np.tanh(exchange_rate**(6)+ exchange_rate**(5)+ exchange_rate**(4)+ exchange_rate**(3)),
        # "sigmoidX_6th":sigmoid(exchange_rate**(7) + exchange_rate**(6)+ exchange_rate**(5)+ exchange_rate**(4)),
        # "sigmoidX_7th": sigmoid(exchange_rate**(6)+ exchange_rate**(5)+ exchange_rate**(4)+ exchange_rate**(3)),
        
        # "dX_dt_squared": np.power(dX_dt, 2),
        # "dX_dt_cubed": np.power(dX_dt, 3),
        # "d2X_dt2_squared": np.power(d2X_dt2, 2),
        # "d2X_dt2_cubed": np.power(d2X_dt2, 3),
        
        # "logX":(np.log(exchange_rate**(6)+ exchange_rate**(5)+ exchange_rate**(4)+ exchange_rate**(3))),
        # "logX2":(np.log(exchange_rate**(2)+ exchange_rate**(3)+ exchange_rate**(1)+ exchange_rate**(0))),
        # "logX3":np.log(exchange_rate**(0)),
        # "logX4":np.log(exchange_rate**(1)),
        # "logX5":np.log(exchange_rate**(2)),
        # "logX6":np.log(exchange_rate**(3)),

        # "sin_X":np.sin(oexchange_rate-exchange_rate),
        # "sin_X-":np.sin(-oexchange_rate+exchange_rate),
        # "cos_X":np.cos(oexchange_rate-exchange_rate),
        # "cos_X-":np.cos(-oexchange_rate+exchange_rate),
        # "xsin_X":exchange_rate*np.sin(oexchange_rate-exchange_rate),
        # "xsin_X-":exchange_rate*np.sin(-oexchange_rate+exchange_rate),
        # "xcos_X":exchange_rate*np.cos(oexchange_rate-exchange_rate),
        # "xcos_X-":exchange_rate*np.cos(-oexchange_rate+exchange_rate),
        # "exp_neg_X": np.exp(-(oexchange_rate-exchange_rate)),
        # "exp_neg_X": np.exp((oexchange_rate-exchange_rate)),

        
        # "exp_neg_dX": np.exp(-dX_dt),
        # "exp_neg_d2X": np.exp(-d2X_dt2),
        # "exp_neg_d3X": np.exp(-d3X_dt3),
        # "XdX_dt": exchange_rate*dX_dt,
        # "dX_dtd2X_dt2": dX_dt*d2X_dt2,
        # "d2X_dt2X": exchange_rate*d2X_dt2,
        # "XdX_dt_squared": exchange_rate*np.power(dX_dt, 2),
        # "XdX_dt_cubed": exchange_rate*np.power(dX_dt, 3),
        # "Xd2X_dt2_squared": exchange_rate*np.power(d2X_dt2, 2),
        # "Xd2X_dt2_cubed": exchange_rate*np.power(d2X_dt2, 3),
        # "Xsin_X": exchange_rate*np.sin(exchange_rate),
        # "Xexp_neg_X": exchange_rate*np.exp(-exchange_rate),
        "high":  np.log(pd.to_numeric(data['High'], errors='coerce')),
        "low": np.log( pd.to_numeric(data['Low'], errors='coerce')),
        "open": np.log( pd.to_numeric(data['Open'], errors='coerce')),
        # "rhigh": ( pd.to_numeric(data['High'], errors='coerce')+(2 * np.random.random(exchange_rate.shape) - 1)*0.01),
        # "rlow": (pd.to_numeric(data['Low'], errors='coerce')+(2 * np.random.random(exchange_rate.shape) - 1)*0.01),
        # "ropen": ( pd.to_numeric(data['Open'], errors='coerce')+(2 * np.random.random(exchange_rate.shape) - 1)*0.01),
        # "odX_dt": odX_dt,
        # "od2X_dt2": od2X_dt2,
        # "od3X_dt3": od3X_dt3*10000000000,
        # "hdX_dt": hdX_dt,
        # "hd2X_dt2": hd2X_dt2,
        # "hd3X_dt3": hd3X_dt3*10000000000,
        # # "hdX_dt_squared": np.power(hdX_dt, 2),
        # # "hdX_dt_cubed": np.power(hdX_dt, 3),
        # "hd2X_dt2_squared": np.power(hd2X_dt2, 2),
        # "hd2X_dt2_cubed": np.power(hd2X_dt2, 3),
        # # "hsin_X": np.sin(hexchange_rate),
        # # "hexp_neg_X": np.exp(-hexchange_rate),
        # # "hexp_neg_dX": np.exp(-hdX_dt),
        # # "hexp_neg_d2X": np.exp(-hd2X_dt2),
        # # "hexp_neg_d3X": np.exp(-hd3X_dt3),
        # # "hXdX_dt": hexchange_rate*hdX_dt,
        # "hdX_dtd2X_dt2": hdX_dt*hd2X_dt2,
        # # "hd2X_dt2X": hexchange_rate*hd2X_dt2,
        # # "hXdX_dt_squared": hexchange_rate*np.power(hdX_dt, 2),
        # # "hXdX_dt_cubed": hexchange_rate*np.power(hdX_dt, 3),
        # # "hXd2X_dt2_squared": hexchange_rate*np.power(hd2X_dt2, 2),
        # # "hXd2X_dt2_cubed": hexchange_rate*np.power(hd2X_dt2, 3),
        
        # "ldX_dt": ldX_dt,
        # "ld2X_dt2": ld2X_dt2,
        # "ld3X_dt3": ld3X_dt3*10000000000,
        # # "ldX_dt_squared": np.power(ldX_dt, 2),
        # # "ldX_dt_cubed": np.power(ldX_dt, 3),
        # "ld2X_dt2_squared": np.power(ld2X_dt2, 2),
        # "ld2X_dt2_cubed": np.power(ld2X_dt2, 3),
        # # "lsin_X": np.sin(lexchange_rate),
        # # "lexp_neg_X": np.exp(-lexchange_rate),
        # # "lexp_neg_dX": np.exp(-ldX_dt),
        # # "lexp_neg_d2X": np.exp(-ld2X_dt2),
        # # "lexp_neg_d3X": np.exp(-ld3X_dt3),
        # # "lXdX_dt": lexchange_rate*ldX_dt,
        # "ldX_dtd2X_dt2": ldX_dt*ld2X_dt2,
        # # "ld2X_dt2X": lexchange_rate*hd2X_dt2,
        # # "lXdX_dt_squared": lexchange_rate*np.power(ldX_dt, 2),
        # # "lXdX_dt_cubed": lexchange_rate*np.power(ldX_dt, 3),
        # # "lXd2X_dt2_squared": lexchange_rate*np.power(ld2X_dt2, 2),
        # # "lXd2X_dt2_cubed": lexchange_rate*np.power(ld2X_dt2, 3),
        
        
        # "mdX_dt": mdX_dt,
        # "md2X_dt2": md2X_dt2,
        # "md3X_dt3": md3X_dt3*10000000000,
        # # "mdX_dt_squared": np.power(mdX_dt, 2),
        # # "mdX_dt_cubed": np.power(mdX_dt, 3),
        # "md2X_dt2_squared": np.power(md2X_dt2, 2),
        # "md2X_dt2_cubed": np.power(md2X_dt2, 3),
        # # "msin_X": np.sin(mexchange_rate),
        # # "mexp_neg_X": np.exp(-mexchange_rate),
        # # "mexp_neg_dX": np.exp(-mdX_dt),
        # # "mexp_neg_d2X": np.exp(-md2X_dt2),
        # # "mexp_neg_d3X": np.exp(-md3X_dt3),
        # # "mXdX_dt": mexchange_rate*mdX_dt,
        # "mdX_dtd2X_dt2": mdX_dt*md2X_dt2,
        # "md2X_dt2X": mexchange_rate*md2X_dt2,
        # "mXdX_dt_squared": mexchange_rate*np.power(mdX_dt, 2),
        # "mXdX_dt_cubed": mexchange_rate*np.power(mdX_dt, 3),
        # "mXd2X_dt2_squared": mexchange_rate*np.power(md2X_dt2, 2),
        # "mXd2X_dt2_cubed": mexchange_rate*np.power(md2X_dt2, 3),
        
        
        
    })
    
    df['y_1'] = np.log(df['X'].shift(1))
 
    df['y_2'] = np.log(df['X'].shift(2))
    # df['y_3'] = np.log(df['X'].shift(3))
    
    # # # Moving Averages
    # df['SMA_2'] = (df['X'].rolling(window=2).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['SMA_3'] = (df['X'].rolling(window=3).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['SMA_4'] = (df['X'].rolling(window=4).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_5'] = (df['X'].rolling(window=5).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_6'] = (df['X'].rolling(window=6).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_7'] = (df['X'].rolling(window=7).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_8'] = (df['X'].rolling(window=8).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_9'] = (df['X'].rolling(window=9).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_10'] = (df['X'].rolling(window=10).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_20'] = (df['X'].rolling(window=20).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_30'] = (df['X'].rolling(window=30).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_40'] = (df['X'].rolling(window=40).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_50'] = (df['X'].rolling(window=50).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_60'] = (df['X'].rolling(window=60).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_70'] = (df['X'].rolling(window=70).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_80'] = (df['X'].rolling(window=80).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_90'] = (df['X'].rolling(window=90).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    df['SMA_100'] = (df['X'].rolling(window=100).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01

    

    # df['1SMA_2'] = (df['X2'].rolling(window=2).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_3'] = (df['X2'].rolling(window=3).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_4'] = (df['X2'].rolling(window=4).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_5'] = (df['X2'].rolling(window=5).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_6'] = (df['X2'].rolling(window=6).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_7'] = (df['X2'].rolling(window=7).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_8'] = (df['X2'].rolling(window=8).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_9'] = (df['X2'].rolling(window=9).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_10'] = (df['X2'].rolling(window=10).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_20'] = (df['X2'].rolling(window=20).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_30'] = (df['X2'].rolling(window=30).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_40'] = (df['X2'].rolling(window=40).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_50'] = (df['X2'].rolling(window=50).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_60'] = (df['X2'].rolling(window=60).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_70'] = (df['X2'].rolling(window=70).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_80'] = (df['X2'].rolling(window=80).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_90'] = (df['X2'].rolling(window=90).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['1SMA_100'] = (df['X2'].rolling(window=100).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01

    # df['2SMA_2'] = (df['X3'].rolling(window=2).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_3'] = (df['X3'].rolling(window=3).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_4'] = (df['X3'].rolling(window=4).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_5'] = (df['X3'].rolling(window=5).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_6'] = (df['X3'].rolling(window=6).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_7'] = (df['X3'].rolling(window=7).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_8'] = (df['X3'].rolling(window=8).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_9'] = (df['X3'].rolling(window=9).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_10'] = (df['X3'].rolling(window=10).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_20'] = (df['X3'].rolling(window=20).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_30'] = (df['X3'].rolling(window=30).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_40'] = (df['X3'].rolling(window=40).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_50'] = (df['X3'].rolling(window=50).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_60'] = (df['X3'].rolling(window=60).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_70'] = (df['X3'].rolling(window=70).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_80'] = (df['X3'].rolling(window=80).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_90'] = (df['X3'].rolling(window=90).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01
    # df['2SMA_100'] = (df['X3'].rolling(window=100).mean())#+(2 * np.random.random(exchange_rate.shape) - 1)*0.01



    # SMA_7 = (df['X'].rolling(window=7).mean())
    # SMA_10 = (df['X'].rolling(window=10).mean())
    # SMA_15 = (df['X'].rolling(window=15).mean())
    # SMA_20 = (df['X'].rolling(window=20).mean())
    # SMA_25 = (df['X'].rolling(window=25).mean())
    # SMA_30 = (df['X'].rolling(window=30).mean())
    # SMA_35 = (df['X'].rolling(window=35).mean())
    # SMA_40 = (df['X'].rolling(window=40).mean()) 
    # SMA_45 = (df['X'].rolling(window=45).mean()) 
    # SMA_50 = (df['X'].rolling(window=50).mean()) 
    # SMA_55 = (df['X'].rolling(window=55).mean()) 
    # SMA_60 = (df['X'].rolling(window=60).mean()) 
    # SMA_65 = (df['X'].rolling(window=65).mean()) 
    # SMA_70 = (df['X'].rolling(window=70).mean()) 

    # df['dSMA_7'] = (SMA_7.diff())
    # df['dSMA_10'] = (SMA_10.diff())
    # df['dSMA_15'] = (SMA_15.diff())
    # df['dSMA_20'] = (SMA_20.diff())
    # df['dSMA_25'] = (SMA_25.diff())
    # df['dSMA_30'] = (SMA_30.diff())
    # df['dSMA_35'] = (SMA_35.diff())
    # df['dSMA_40'] = (SMA_40.diff())
    # df['dSMA_45'] = (SMA_45.diff())
    # df['dSMA_50'] = (SMA_50.diff())
    # df['dSMA_55'] = (SMA_55.diff())
    # df['dSMA_60'] = (SMA_60.diff())
    # df['dSMA_65'] = (SMA_65.diff())
    # df['dSMA_70'] = (SMA_70.diff())
    
    # df['SMA_30'] = sigmoid(df['X'].rolling(window=2).mean()) 
    # df['EMA_50'] = (df['X'].ewm(span=50).mean())

    # Relative Strength Index (RSI)
    # def calculate_rsi(prices, window=14):
    #     delta = prices.diff()
    #     gain = delta.mask(delta < 0, 0)
    #     loss = -delta.mask(delta > 0, 0)
    #     avg_gain = gain.rolling(window).mean()
    #     avg_loss = loss.rolling(window).mean()
    #     rs = avg_gain / avg_loss
    #     rsi = 100 - (100 / (1 + rs))
    #     return rsi

    # df['RSI_14'] = calculate_rsi(df['X'], window=14)
    # df['RSI_14']=df['RSI_14'].diff()
    # # Bollinger Bands
    # def calculate_bollinger_bands(prices, window=20, num_std=2):
    #     rolling_mean = prices.rolling(window).mean()
    #     rolling_std = prices.rolling(window).std()
    #     upper_band = rolling_mean + (rolling_std * num_std)
    #     lower_band = rolling_mean - (rolling_std * num_std)
    #     return upper_band, lower_band

    # # df['BB_upper'], df['BB_lower'] = (calculate_bollinger_bands(df['X'], window=20, num_std=2))
    # # df['BB_upper'], df['BB_lower']= (df['BB_upper'].diff()*10000000000 *10), (random.random()+df['BB_lower'].diff()*10000000000)
    # # # # MACD (Moving Average Convergence Divergence)
    # exp_ma_12 = df['X'].ewm(span=12).mean() 
    # exp_ma_26 = df['X'].ewm(span=26).mean() 
    # df['MACD_line'] = exp_ma_12 - exp_ma_26
    # df['MACD_signal'] = df['MACD_line'].ewm(span=9).mean()
    # df['MACD_histogram'] = df['MACD_line'] - df['MACD_signal']

    # df['MACD_line'] = df['MACD_line'].diff()*10000000000 
    # df['MACD_signal'] = df['MACD_signal'].diff()*10000000000 
    # df['MACD_histogram'] = df['MACD_histogram'].diff()*10000000000 


    # # Stochastic Oscillator
    # def calculate_stochastic_oscillator(prices, window=14):
    #     lowest_low = prices.rolling(window).min()
    #     highest_high = prices.rolling(window).max()
    #     stochastic_k = ((prices - lowest_low) / (highest_high - lowest_low)) * 100
    #     stochastic_d = stochastic_k.rolling(3).mean()
    #     return stochastic_k, stochastic_d

    # df['%K'], df['%D'] = calculate_stochastic_oscillator(df['X'], window=14)
    # df['%K'], df['%D'] = calculate_stochastic_oscillator(df['X'], window=20)
    # df['%K'], df['%D'] = calculate_stochastic_oscillator(df['X'], window=40)
    # df['%K'], df['%D'] = calculate_stochastic_oscillator(df['X'], window=60)
    # df['%K'], df['%D'] = calculate_stochastic_oscillator(df['X'], window=80)
    # df['%K'], df['%D'] = calculate_stochastic_oscillator(df['X'], window=80)

    # Volume
    df['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
    # df['dVolume'] =df['Volume'].diff() 
    # df['ddVolume'] =df['Volume'].diff().diff() 
    # df['dddVolume'] =df['Volume'].diff().diff().diff() 
    # df['ddddVolume'] =df['Volume'].diff().diff().diff().diff() 
    # df['dddddVolume'] =df['Volume'].diff().diff().diff().diff().diff() 
    # df['ddddddVolume'] =df['Volume'].diff().diff().diff().diff().diff().diff() 
    # df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean() 
    # df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10'] 
    # Volume_SMA_2 = (df['Volume'].rolling(window=2).mean())
    # Volume_SMA_5 = (df['Volume'].rolling(window=3).mean())
    # Volume_SMA_10 = (df['Volume'].rolling(window=4).mean())
    # Volume_SMA_15 = (df['Volume'].rolling(window=5).mean())
    # Volume_SMA_20 = (df['Volume'].rolling(window=6).mean())
    # Volume_SMA_25 = (df['Volume'].rolling(window=7).mean())
    # Volume_SMA_30 = (df['Volume'].rolling(window=8).mean())
    # Volume_SMA_35 = (df['Volume'].rolling(window=9).mean())
    # Volume_SMA_40 = (df['Volume'].rolling(window=10).mean())
    # Volume_SMA_45 = (df['Volume'].rolling(window=11).mean())
    # Volume_SMA_50 = (df['Volume'].rolling(window=12).mean())
    # Volume_SMA_55 = (df['Volume'].rolling(window=13).mean())
    # Volume_SMA_60 = (df['Volume'].rolling(window=14).mean())
    # Volume_SMA_65 = (df['Volume'].rolling(window=15).mean())
    # Volume_SMA_70 = (df['Volume'].rolling(window=16).mean())
    # df = df.drop('Volume', axis=1)
    
    # df['Volume_dSMA_2'] = (Volume_SMA_2.diff()) 
    # df['Volume_dSMA_5'] = (Volume_SMA_5.diff()) 
    # df['Volume_dSMA_10'] = (Volume_SMA_10.diff()) 
    # df['Volume_dSMA_15'] = (Volume_SMA_15.diff()) 
    # df['Volume_dSMA_20'] = (Volume_SMA_20.diff()) 
    # df['Volume_dSMA_25'] = (Volume_SMA_25.diff()) 
    # df['Volume_dSMA_30'] = (Volume_SMA_30.diff()) 
    # df['Volume_dSMA_35'] = (Volume_SMA_35.diff()) 
    # df['Volume_dSMA_40'] = (Volume_SMA_40.diff()) 
    # df['Volume_dSMA_45'] = (Volume_SMA_45.diff()) 
    # df['Volume_dSMA_50'] = (Volume_SMA_50.diff()) 
    # df['Volume_dSMA_55'] = (Volume_SMA_55.diff()) 
    # df['Volume_dSMA_60'] = (Volume_SMA_60.diff()) 
    # df['Volume_dSMA_65'] = (Volume_SMA_65.diff()) 
    # df['Volume_dSMA_70'] = (Volume_SMA_70.diff()) 
    # # 
    # df['Volume_ddSMA_2'] = (df['Volume_dSMA_2'].diff()) 
    # df['Volume_ddSMA_5'] = (df['Volume_dSMA_5'].diff()) 
    # df['Volume_ddSMA_10'] = (df['Volume_dSMA_10'].diff()) 
    # df['Volume_ddSMA_15'] = (df['Volume_dSMA_15'].diff()) 
    # df['Volume_ddSMA_20'] = (df['Volume_dSMA_20'].diff()) 
    # df['Volume_ddSMA_25'] = (df['Volume_dSMA_25'].diff()) 
    # df['Volume_ddSMA_30'] = (df['Volume_dSMA_30'].diff()) 
    # df['Volume_ddSMA_35'] = (df['Volume_dSMA_35'].diff()) 
    # df['Volume_ddSMA_40'] = (df['Volume_dSMA_40'].diff()) 
    # df['Volume_ddSMA_45'] = (df['Volume_dSMA_45'].diff()) 
    # df['Volume_ddSMA_50'] = (df['Volume_dSMA_50'].diff()) 
    # df['Volume_ddSMA_55'] = (df['Volume_dSMA_55'].diff()) 
    # df['Volume_ddSMA_60'] = (df['Volume_dSMA_60'].diff()) 
    # df['Volume_ddSMA_65'] = (df['Volume_dSMA_65'].diff()) 
    # df['Volume_ddSMA_70'] = (df['Volume_dSMA_70'].diff()) 


    # SMA_7 = (df['X'].rolling(window=7).mean())
    # SMA_10 = (df['X'].rolling(window=10).mean())
    # SMA_15 = (df['X'].rolling(window=15).mean())
    # SMA_20 = (df['X'].rolling(window=20).mean())
    # SMA_25 = (df['X'].rolling(window=25).mean())
    # SMA_30 = (df['X'].rolling(window=30).mean())
    # SMA_35 = (df['X'].rolling(window=35).mean())
    # SMA_40 = (df['X'].rolling(window=40).mean())
    # SMA_45 = (df['X'].rolling(window=45).mean())
    
    # df['dSMA_7'] = np.tanh(SMA_7.diff())
    # df['dSMA_10'] = np.tanh(SMA_10.diff())
    # df['dSMA_15'] = np.tanh(SMA_15.diff())
    # df['dSMA_20'] = np.tanh(SMA_20.diff())
    # df['dSMA_25'] = np.tanh(SMA_25.diff())
    # df['dSMA_30'] = np.tanh(SMA_30.diff())
    # df['dSMA_35'] = np.tanh(SMA_35.diff())
    # df['dSMA_40'] = np.tanh(SMA_40.diff())
    # df['dSMA_45'] = np.tanh(SMA_45.diff())


    # # Candlestick Patterns
    # # candle_patterns = CandlePatterns(df['open'], df['high'], df['low'], df['close'])
    # # df['Doji'] = candle_patterns.is_doji()

    # Fibonacci Levels
    # def calculate_fibonacci_38_2(prices):
    #     max_price = max(prices)
    #     min_price = min(prices)
    #     diff = max_price - min_price
    #     fib_38_2 = min_price + 0.382 * diff
    #     return fib_38_2
    
    # def calculate_fibonacci_50(prices):
    #     max_price = max(prices)
    #     min_price = min(prices)
    #     diff = max_price - min_price
    #     fib_50 = min_price + 0.5 * diff
    #     return fib_50
    
    # def calculate_fibonacci_61_8(prices):
    #     max_price = max(prices)
    #     min_price = min(prices)
    #     diff = max_price - min_price
    #     fib_61_8 = min_price + 0.618 * diff
    #     return fib_61_8
    
    # # Choose a window size
    # window_size = 20
    
    # Use rolling window and apply function
    # df['Fib_38.2'] = df['X'].rolling(window_size).apply(calculate_fibonacci_38_2, raw=True) 
    # df['Fib_50'] = df['X'].rolling(window_size).apply(calculate_fibonacci_50, raw=True) 
    # df['Fib_61.8'] = df['X'].rolling(window_size).apply(calculate_fibonacci_61_8, raw=True) 

    # df['dFib_38.2'] =  df['Fib_38.2'].diff()*10000000000
    # df['dFib_50'] = df['Fib_50'].diff()*10000000000
    # df['dFib_61.8'] = df['Fib_61.8'].diff()*10000000000

    # df['ddFib_38.2'] =  df['Fib_38.2'].diff().diff()*10000000000
    # df['ddFib_50'] = df['Fib_50'].diff().diff()*10000000000
    # df['ddFib_61.8'] = df['Fib_61.8'].diff().diff()*10000000000
    # Sentiment Analysis
    # df['Sentiment'] = df['News'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Economic Indicators
    # df['GDP_growth_rate'] = economic_data['GDP_growth_rate']
    
    
    # Calculate the logarithmic returns
    df['log_returns'] = np.log(df['X'] / df['X'].shift(1))
    
    # Estimate drift (μ) and volatility (σ) from the logarithmic returns
    drift = df['log_returns'].mean()
    volatility = df['log_returns'].std()
    
    # Define the time interval
    dt = 1  # Assuming daily observations, dt = 1 day
    
    # Define the parameters for the trend term, the quadratic term, and the jump component
    alpha = 0.01  # Trend coefficient
    beta = 0.001  # Quadratic coefficient
    gamma= 0.001
    delta = 0.005  # Open price coefficient
    epsilon = 0.005  # Close price coefficient
    lambda_ = 1.1  # Jump intensity
    jump_mean = 0.05  # Mean of the jump size
    jump_std = 0.1  # Standard deviation of the jump size
    
    # Compute the terms dS(t), μS(t)dt, and σS(t)dW(t) using the estimated parameters
    df['dS1'] = df['X'].diff()  # dS(t) is the difference between consecutive prices
    df['dS2'] = df['X'].diff()  # dS(t) is the difference between consecutive prices
    df['dS3'] = df['X'].diff()  # dS(t) is the difference between consecutive prices
    df['dS4'] = df['X'].diff()  # dS(t) is the difference between consecutive prices
    df['dS5'] = df['X'].diff()  # dS(t) is the difference between consecutive prices
    df['dS6'] = df['X'].diff()  # dS(t) is the difference between consecutive prices

    df['mu_S_dt'] = (drift + alpha * df.index.dayofyear + beta * df['X']**2) * df['X'] * dt  # (μ + αt + βS(t)^2)S(t)dt
    df['mu_S_dt2'] = (drift + alpha * df.index.dayofyear) * df['X'] * dt  # (μ + αt + βS(t)^2)S(t)dt
    df['mu_S_dt3'] = (drift + alpha * df.index.dayofyear+ gamma * df['Volume']) * df['X'] * dt  # (μ + αt + βS(t)^2)S(t)dt
    df['mu_S_dt4'] = (drift + alpha * df.index.dayofyear+ beta * df['X']**2+ gamma * df['Volume']) * df['X'] * dt  # (μ + αt + βS(t)^2)S(t)dt
    df['mu_S_dt5'] = (drift + alpha * df.index.dayofyear + beta * df['X']**2 + gamma * df['Volume'] + delta * df['open'] + epsilon * df['X']) * df['X'] * dt  # (μ + αt + βS(t)^2 + γV(t) + δO(t) + εC(t))S(t)dt
    df['mu_S_dt6'] = (drift + alpha * df.index.dayofyear + beta * df['X']**2 + gamma * df['Volume'] + delta * df['high'] + epsilon * df['low']) * df['X'] * dt  # (μ + αt + βS(t)^2 + γV(t) + δO(t) + εC(t))S(t)dt

    
    
    df['sigma_S_dW'] = volatility * df['X'] * np.random.normal(0, np.sqrt(dt), size=len(df))  # σS(t)dW(t)
    
    # Compute the jump component J(t)dN(t) using a Poisson process and a jump size distribution
    poisson_process = np.random.poisson(lambda_ * dt, size=len(df))  # Poisson process
    jump_sizes = np.random.normal(jump_mean, jump_std, size=len(df))  # Jump sizes
    df['jump_component'] = jump_sizes * poisson_process
    
    # Compute the final SDE dS(t) = (μ + αt + βS(t)^2)S(t)dt + σS(t)dW(t) + J(t)dN(t)
    df['dS1'] += df['mu_S_dt'] + df['sigma_S_dW']
    df['dS2'] += df['mu_S_dt2'] + df['sigma_S_dW'] + df['jump_component']
    df['dS3'] += df['mu_S_dt2'] + df['sigma_S_dW'] + df['jump_component']
    df['dS4'] += df['mu_S_dt4'] + df['sigma_S_dW']+ df['jump_component']
    df['dS5'] += df['mu_S_dt5'] + df['sigma_S_dW']+ df['jump_component']
    df['dS6'] += df['mu_S_dt6'] + df['sigma_S_dW']+ df['jump_component']


    # df = df.drop('open', axis=1)
    # df = df.drop('high', axis=1)
    # df = df.drop('low', axis=1)
    # df = df.drop('Volume', axis=1)
    # df = df.drop('mu_S_dt', axis=1)
    # df = df.drop('mu_S_dt2', axis=1)
    # df = df.drop('mu_S_dt3', axis=1)
    # df = df.drop('mu_S_dt4', axis=1)
    # df = df.drop('mu_S_dt5', axis=1)
    # df = df.drop('mu_S_dt6', axis=1)
    df = df.drop('X1', axis=1)

    # Print the resulting DataFrame
    
    # print(df['dS'])
    
    return df