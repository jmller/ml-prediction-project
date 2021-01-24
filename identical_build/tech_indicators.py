#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Indicators;
Formulas of technical indicators can be found here
They may be called by the continuous layer
If the technical Indicators are deviating from the underlying paper, 
the underlying sources can be found in the comment above the function
"""
import pandas as pd
import numpy as np

###
#Technical Indicators and Data Processing
###

#Simple n(=10)-day Moving Average using PANDAS
def MA(timeseries, n = 10):
    return timeseries.rolling(n).mean()

#Weighted n(=10)-day Moving Average
#https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9
def WMA(timeseries, n = 10):
    weights = np.arange(1,n+1) #this creates an array with integers 1 to 10 included
    wma_n = timeseries.rolling(n).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    return wma_n

#Exponential Moving Average
#https://steemit.com/trading/@jwyles/ema-ema-and-dema-how-to-calculate-using-python-and-spreadsheets
def EMA(timeseries, k = 10):
    ema =[0]*10 #NA für die ersten 10 Tage
    alpha = 2/(k+1)
    ema.append(timeseries.loc[0:k-1, 'Close'].sum()/k) 
    for x in range(k+1, len(timeseries)):
        ema.append(ema[x-1]+alpha*(timeseries.loc[x, 'Close'] - ema[x-1]))
    ema = pd.DataFrame(ema)
    return ema

#Momentum
def MOMENTUM (timeseries, n = 10):
    return timeseries.rolling(n).apply(lambda prices: prices.iloc[-1]-prices.iloc[0])

#Stochastic K%
def STOCHASTIC_K(timeseries, n = 10):
    Zaehler = timeseries['Close'] - timeseries['Low'].rolling(n).min()
    Nenner = timeseries['High'].rolling(n).max() - timeseries['Low'].rolling(n).min()
    return 100*(Zaehler/Nenner)

#Stochastic D%
#https://www.investopedia.com/terms/s/stochasticoscillator.asp#:~:text=The%20Formula%20For%20The%20Stochastic%20Oscillator%20Is&text=The%20%22fast%22%20stochastic%20indicator%20is,period%20moving%20average%20of%20%25K.&text=Transaction%20signals%20are%20created%20when,which%20is%20called%20the%20%25D.
#n = 3, standard
def STOCHASTIC_D(timeseries, n = 3, n_stochk = 10):
    return (STOCHASTIC_K(timeseries, n_stochk).rolling(n).mean())
#https://www.investopedia.com/terms/s/stochasticoscillator.asp

#Calculation derived from an excel-sheet
#Relative-Strength-Index   
#https://www.macroption.com/rsi-calculation/
def RSI(timeseries, n = 10):
    change = timeseries['Close']-timeseries['Close'].shift(1)
    gain_bool = change >= 0 
    loss_bool = change < 0
    g1 = gain_bool*change
    l1 = np.abs(loss_bool*change)
    avg_gain = g1.rolling(n).mean()
    avg_loss = l1.rolling(n).mean()
    rs = avg_gain/avg_loss
    result = 100-100/(1+rs)
    return result

#Using the following MACD-Strategy:
#https://www.diva-portal.org/smash/get/diva2:721625/FULLTEXT01.pdf
#https://towardsdatascience.com/implementing-macd-in-python-cc9b2280126a
#Used the code from towardsdatascience
def MACD(timeseries, n = 10):
    exp1 = timeseries['Close'].ewm(span=12, adjust=False).mean()
    exp2 = timeseries['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()
    crossover = macd - exp3 #if positive then bullish Crossover
    return crossover

#Using Formulas from:
#https://www.investopedia.com/terms/w/williamsr.asp
def Larry(timeseries, n = 14):
    Zaehler = timeseries['High'].rolling(n).max() - timeseries['Close']
    Nenner = timeseries['High'].rolling(n).max() - timeseries['Low'].rolling(n).min()
    return 100*(Zaehler/Nenner)

#Using Formulas from:
#https://www.investopedia.com/articles/trading/08/accumulation-distribution-line.asp
#https://www.investopedia.com/terms/a/accumulationdistribution.asp
def ADOSCI(timeseries, n = 14):
    Zaehler = timeseries['Close'] - timeseries['Low'].rolling(n).min() - (timeseries['High'].rolling(n).max() - timeseries['Close'])  #(C−L)−(H−C)
    Nenner = timeseries['High'].rolling(n).max() - timeseries['Low'].rolling(n).min() #H - L
    return Zaehler/Nenner

#Using Formulas from:
#https://www.investopedia.com/terms/c/commoditychannelindex.asp
#https://en.wikipedia.org/wiki/Commodity_channel_index
def CCI(timeseries, n = 20):
    #Variante 2: M = (timeseries['High'].rolling(n).max() + timeseries['Low'].rolling(n).min() + timeseries['Close'].rolling(n).mean())/3
    M = (timeseries['High'] + timeseries['Low'] + timeseries['Close'])/3
    Zaehler = M - MA(timeseries['Close'], n)
    Nenner = 0.015 * np.abs((M - MA(timeseries['Close'], n)).rolling(n).mean())
    return Zaehler/Nenner