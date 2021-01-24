#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous Layer;
Processes the output of the technical indicators

"""
import pandas as pd
import tech_indicators as ti

def cont_label(timeseries):
    return timeseries['Close'] - timeseries['Close'].shift()
###
# Calling the technical indicators; passing information; further processing of the technical data
###
def cont_layer(timeseries):
    pred_cont_ma = timeseries['Close'] - ti.MA(timeseries['Close'], n =10)
    pred_cont_wma = timeseries['Close'] - ti.WMA(timeseries['Close'], n =10)
    pred_cont_mom = ti.MOMENTUM (timeseries['Close'], n = 10)
    pred_cont_stochk = ti.STOCHASTIC_K(timeseries) - ti.STOCHASTIC_K(timeseries).shift(1)
    pred_cont_stochd = ti.STOCHASTIC_D(timeseries) - ti.STOCHASTIC_D(timeseries).shift(1)
    pred_cont_larry = ti.Larry(timeseries, n = 14) - ti.Larry(timeseries, n = 14).shift(1)
    pred_cont_rsi = ti.RSI(timeseries, n =10)
    pred_cont_macd = ti.MACD(timeseries, n = 10) # Deviating from the paper the Crossover is calculated differently (see: towardsdatascience)
    pred_cont_cci = ti.CCI(timeseries, n=20)
    pred_cont_adosci = ti.ADOSCI(timeseries, n = 14) - ti.ADOSCI(timeseries, n = 14).shift(1)
    return_df = pd.concat([pred_cont_ma, 
                          pred_cont_wma, 
                          pred_cont_mom, 
                          pred_cont_stochk,
                          pred_cont_stochd,
                          pred_cont_larry,
                          pred_cont_macd,
                          pred_cont_cci,
                          pred_cont_adosci,
                          pred_cont_rsi], axis=1)
    return_df.columns = ['MA', 'WMA', 'MOM', 'STOCHK', 'STOCHD', 'LARRY', 'MACD', 'CCI', 'ADOSCI', 'RSI']
    return return_df