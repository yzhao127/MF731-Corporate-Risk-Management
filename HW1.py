# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:26:50 2019

@author: yyang
"""
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
#import pyflux as pf

# MF 731
# HW1 - Part I
# 3
data = pd.read_csv("C:\\Users\\yyang\\OneDrive\\Documents\\Documents\\BU\\Courses\\19Fall\\MF731\\HW\\HW1\\Nasdaq_Data.csv", header=0)
data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)

delta = 1/252
n = 100

log_ret = np.log(data['Closing Value']).diff().dropna()

# MA
vol_ma = pd.Series(index=log_ret.index[n:])
for i in range(len(log_ret)-n): 
    vol_ma.iloc[i] = log_ret.iloc[i: i+n].var()/delta
    
# EWMA
lam1 = 0.94
lam2 = 0.97

vol0 = log_ret.iloc[0: 0+n].var()
vol_ewma1 = pd.Series(index=vol_ma.index)
vol_ewma2 = pd.Series(index=vol_ma.index)
vol_ewma1.iloc[0] = vol0
vol_ewma2.iloc[0] = vol0

log_ret_square = log_ret[n:]**2

for i in range(1, len(log_ret)-n, 1):
    vol_ewma1.iloc[i] = lam1*vol_ewma1.iloc[i-1]+(1-lam1)*log_ret_square.iloc[i-1]
    vol_ewma2.iloc[i] = lam2*vol_ewma2.iloc[i-1]+(1-lam2)*log_ret_square.iloc[i-1]
    
vol_ewma1 = vol_ewma1/delta
vol_ewma2 = vol_ewma2/delta

vol_ma.rename('vol_ma', inplace=True)
vol_ewma1.rename('vol_ewma94', inplace=True)
vol_ewma2.rename('vol_ewma97', inplace=True)

vol = pd.concat([vol_ma, vol_ewma1, vol_ewma2], axis=1)

vol.plot()

# GARCH
df_garch = log_ret.loc['2005-01-04': '2013-12-31']*np.sqrt(1/delta)
garch = arch_model(df_garch, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='Normal')
results = garch.fit()
print(results.summary())
results.plot()

alpha0 = results.params.omega
alpha1 = results.params.loc['alpha[1]']
beta1 = results.params.loc['beta[1]']

#model = pf.GARCH(df_garch.values,p=1,q=1)
#x = model.fit()
#print(x.summary(transformed=False))

Vl = alpha0/(1-alpha1-beta1)

df_actual = log_ret.loc['2014']
V_a = df_actual.var()/delta

zt = np.random.normal(0, 1, (251, 50))
#simu_v = pd.DataFrame(columns=range(50), index=range(252))
simu_v = []
simu_v = [[Vl] * 50]
for i in range(1, 252, 1): 
    simu_v += [alpha0 + simu_v[i-1] * (alpha1*zt[i-1]**2+beta1)]

simu_v[1: ] = [simu_v[i].tolist() for i in range(1, 252, 1)]
df_simu_v = pd.DataFrame(simu_v)

plt.figure(figsize=(20,10))
plt.plot(df_simu_v)
plt.title('One-year sample path for annualized volatility')
plt.xlabel("Annualized vol.")
plt.ylabel("Date")





