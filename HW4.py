# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:14:20 2019

@author: yyang
"""

import numpy as np
import pandas as pd

# HW4
# 2
alpha = 0.95
M = 100
lam = 0.97
theta = 0.97
K = 10
mkt_caps = np.array([448.77, 575.11])

data = pd.read_csv("C:\\Users\\yyang\\OneDrive\\Documents\\Documents\\BU\\Courses\\19Fall\\MF731\\HW\\HW4\\MSFT_AAPL_Log_Returns.csv", names=['Date', 'MSFT', 'AAPL'])
data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)

n_ret = len(data)
port_value = 1000000
dollar_pos = port_value*mkt_caps/np.sum(mkt_caps)

sigma_hat = data.iloc[0:M].cov()
mu_hat = data.iloc[0:M].mean()

for i in range(M, n_ret, 1): 
    sigma_hat = theta*sigma_hat+(1-theta)*((data.iloc[i]-mu_hat).values.reshape(2,1))*((data.iloc[i]-mu_hat).values.reshape(1,2))
    mu_hat = lam*mu_hat+(1-lam)*data.iloc[i]
    

    
    
