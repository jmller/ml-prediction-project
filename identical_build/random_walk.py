#Erzeugung kuenstlicher Daten
import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt

iNumObs = 2000
iNumVars = 7
mX = np.random.randn(iNumObs,iNumVars)

def gen_uncorr_data(iNumObs, iNumVars, mX):
    #unkorrelierte Daten
    
    #Date (X0)
    #mX[:, 0] = pd.date_range(start='1/1/2018', periods = iNumObs)
    mX[:, 0] = np.linspace(0, iNumObs-1, iNumObs)
    
    #Open (X1)
    mX[:, 1] = 1000
    sigma_x1 = 10
    for x in range(1, iNumObs):
        mX[x, 1] = mX[x-1, 1] + np.random.randn()*sigma_x1
    
    #Close (X4)
    sigma_x4 = 0.2
    for x in range(0, iNumObs-1):
        mX[x, 4] = mX[x+1, 1] + np.random.randn()*sigma_x4
    mX[-1, 4] = mX[-1, 1] + np.random.randn()*sigma_x4
    
    #High (X2)
    sigma_x2 = 0.2
    mX[:, 2] = np.maximum(mX[:, 1], mX[:, 4]) + abs(np.random.randn()*sigma_x2)
    
    #Low (X3)
    sigma_x3 = 0.2
    mX[:, 3] = np.minimum(mX[:, 1], mX[:, 4]) - abs(np.random.randn()*sigma_x3)
    
    #Shares tradet (X5) (unkorreliert)
    mu = 10000
    sigma_x5 = mu/2
    mX[:, 5] = np.random.randn(iNumObs)*sigma_x5 + mu
    
    #Turnover (X6) (unkorreliert)
    mu = 2000
    sigma_x6 = mu/5
    mX[:, 6] = np.random.randn(iNumObs)*sigma_x6 + mu
    
    return mX

#Bei Bedarf knnen weitere Abhngigkeiten zwischen den Features erzeugt werden.

test = gen_uncorr_data(iNumObs, iNumVars, mX)
test_pd = pd.DataFrame(data = test[:,:])
test_pd= test_pd.round(2)
test_pd.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Shares Traded',	'Turnover (Rs. Cr)']
test_pd.to_csv("randomwalk.csv")

plt.plot(test_pd['Close'])
plt.ylabel('Kurs')
plt.xlabel('Samples')
