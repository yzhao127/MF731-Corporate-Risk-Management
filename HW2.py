# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:45:16 2019

@author: yyang
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model

# MF 731
# HW2 - Part I
# 3

data = pd.read_csv("C:\\Users\\yyang\\OneDrive\\Documents\\Documents\\BU\\Courses\\19Fall\\MF731\\HW\\HW2\\CAT_TSLA_Data.csv", header=None, names=['Date', 'cat', 'tsla'])
data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)

delta = 1/252
log_ret = np.log(data).diff().dropna()

lam = 0.97
theta = 0.97

ret_avg = log_ret.mean()
ret_ewma = pd.DataFrame(index=data.index, columns=data.columns)
ret_ewma.iloc[0] = ret_avg

cov_avg = log_ret.cov()
cov_ewma = [cov_avg.values]

for i in range(len(log_ret)):
    ret_ewma.iloc[i+1] = lam*ret_ewma.iloc[i]+(1-lam)*log_ret.iloc[i]
    ret_dif = log_ret.iloc[i]-ret_ewma.iloc[i]
    cov_ewma += [(theta*cov_ewma[i]+(1-theta)*ret_dif.values.reshape(2,1)*ret_dif.values.reshape(1,2)).astype(float)]

w_c = 82.52/(82.52+51.46)
w_t = 51.46/(82.52+51.46)

ret_port = log_ret['cat']*w_c+log_ret['tsla']*w_t
var_emp = ret_port.quantile(0.95)*1000000

mean_samp = [ret_ewma.iloc[-1].values[0], ret_ewma.iloc[-1].values[1]]
cov_samp = [[cov_ewma[-1][0][0], cov_ewma[-1][0][1]], [cov_ewma[-1][1][0], cov_ewma[-1][1][1]]]
c, t = np.random.multivariate_normal(mean_samp, cov_samp, 25000).T
ret_port_mc = c*w_c+t*w_t
var_mc = np.quantile(ret_port_mc, 0.95) *1000000

mean_ar = np.asarray(mean_samp)
cov_ar = np.asarray(cov_samp)
weight_vec = np.array([w_c, w_t])
var_an = (-weight_vec.dot(mean_ar)+np.sqrt(weight_vec.dot(cov_ar.dot(weight_vec)))*norm.ppf(0.95))*1000000

# Part II
# 1
alpha = 0.95
t = 0
K = 10
delta = 1/252
T = 0.25
s0 = 158.12
k = 170
r = 0.0132
kt = int((T-t)/delta)
sigma = 0.2214
mu = 0.15475
n = 10000


x = np.random.normal((mu-sigma**2/2)*delta, sigma*delta**0.5, (kt, n))
s = pd.DataFrame(index=range(kt+1), columns=range(n))
s.iloc[0] = s0
for i in range(len(s.index)-1):
    s.iloc[i+1] = s.iloc[i]*np.exp(x[i])

d1 = pd.DataFrame(index=range(kt+1), columns=range(n))
d2 = pd.DataFrame(index=range(kt+1), columns=range(n))
c_bs = pd.DataFrame(index=range(kt+1), columns=range(n))
for i in range(kt+1):
    d1.iloc[i] = ((s.iloc[i]/k).map(lambda x: np.log(x))+((r+sigma**2/2)*(T-i*delta)))/(sigma*(T-i*delta)**0.5)
    d2.iloc[i] = d1.iloc[i]-sigma*(T-i*delta)**0.5
    c_bs.iloc[i] = s.iloc[i]*d1.iloc[i].map(lambda x: norm.cdf(x))-k*np.exp(-r*i*delta)*d2.iloc[i].map(lambda x: norm.cdf(x))

ht = pd.DataFrame(index=range(kt+1), columns=range(n))
for i in range(kt+1):
    ht.iloc[i] = d1.iloc[i].map(lambda x: norm.cdf(x)) 

vt = s.iloc[0:-1]*ht.iloc[0:-1]-c_bs.iloc[0:-1]
vt_next_ar = s.iloc[1:].values*ht.iloc[0:-1].values-c_bs.iloc[1:].values
vt_next = pd.DataFrame(index=range(kt), columns=range(n), data=vt_next_ar)
vt_reb = s.iloc[1:]*ht.iloc[1:]-c_bs.iloc[1:]
vt_reb.reset_index(drop=True, inplace=True)
yt = vt_reb - vt_next
vt_real_ar = s.iloc[1:].values*ht.iloc[0:-1].values-c_bs.iloc[1:].values+yt.values*np.exp(r*delta)
vt_real = pd.DataFrame(vt_real_ar)
df_loss = -vt_real.diff().dropna()
df_loss.reset_index(drop=True, inplace=True)

# One day VaR
VaR_every_day = []
for i in df_loss.columns:
    VaR_every_day.append(df_loss[i].quantile(0.95))

one_day_VaR = np.mean(VaR_every_day)

df_loss_10 = -vt_real.diff(periods=10).dropna()
VaR_every_10days = []
for i in df_loss_10.columns:
    VaR_every_10days.append(df_loss_10[i].quantile(0.95))

# Ten days VaR
ten_day_VaR = np.mean(VaR_every_10days)
# Ten days VaR
ten_day_VaR2 = one_day_VaR*(10**0.5)

#Print Results
dict_result = {"One day VaR": one_day_VaR, "Ten days VaR": ten_day_VaR, "Ten days VaR by Agg": ten_day_VaR2}
df_result = pd.DataFrame(dict_result, index=['Value'])
df_result

# 2
df_sp = pd.read_csv("C:\\Users\\yyang\\OneDrive\\Documents\\Documents\\BU\\Courses\\19Fall\\MF731\\HW\\HW2\\SP_Prices.csv", header=None, names=['price'])
df_sp.index = pd.to_datetime(df_sp.index)
sp_ret = np.log(df_sp).diff().dropna()

N = len(sp_ret)
M = 1010
alpha = 0.95
beta = 0.05
lam = 0.97
theta = 0.97

ptf_loss = 1 - np.exp(sp_ret)

# GARCH
mu0 = ptf_loss.iloc[0: M].values.mean()*252
vol0 = ptf_loss.iloc[0: M].values.var()*np.sqrt(252)

alpha0 = []
alpha1 = []
beta1 = []
mu = [mu0]
vol = [vol0]
VaR_GARCH = []
exc_GARCH = 0
for i in range(N-M):
    garch = arch_model(ptf_loss.iloc[i: i+M]*252, vol='Garch', p=1, o=0, q=1, dist='Normal')
    results = garch.fit()
    alpha0.append(results.params.omega)
    alpha1.append(results.params.loc['alpha[1]'])
    beta1.append(results.params.loc['beta[1]'])
    mu.append(lam*mu[i]+(1-lam)*ptf_loss.iloc[i+M-1].values*252)
    vol.append(alpha0[i]+alpha1[i]*(ptf_loss.iloc[i+M-1].values*252-mu[i])**2+beta1[i]*vol[i])
    VaR_GARCH.append(mu[-1]+np.sqrt(vol[-1])*norm.ppf(alpha))
    exc_GARCH = exc_GARCH + sum(ptf_loss.iloc[i+M]*252>VaR_GARCH[-1][0])

# empirical distribution
VaR_emp = []
exc_emp = 0
for i in range(N-M): 
    VaR_emp.append(np.quantile(ptf_loss.iloc[i: i+M].values, alpha)*252)
    exc_emp = exc_emp + sum(ptf_loss.iloc[i+M]*252>VaR_emp[-1])
    
# EWMA
exc_ewma = 0 
mu_ewma = [mu0]
vol_ewma = [vol0]
VaR_ewma = []
for i in range(N-M): 
    mu_ewma.append(lam*mu[i]+(1-lam)*ptf_loss.iloc[i+M-1].values*252)
    vol_ewma.append(theta*vol_ewma[i]+(1-theta)*(ptf_loss.iloc[i+M-1].values*252-mu_ewma[i])**2)
    VaR_ewma.append(mu_ewma[-1]+np.sqrt(vol_ewma[-1])*norm.ppf(alpha))
    exc_ewma = exc_ewma + sum(ptf_loss.iloc[i+M]*252>VaR_ewma[-1][0])

# Confidence Interval
m = N-M
CI_low = m * (1-alpha) - norm.ppf(1-beta/2) * np.sqrt(m*alpha*(1-alpha))
CI_high = m * (1-alpha) + norm.ppf(1-beta/2) * np.sqrt(m*alpha*(1-alpha))

# Print Result
# 1
print("===============================================================")
print("The result of Q.1: ")
print(df_result)

# 2
print("\n===============================================================")
print("\nThe result of Q.2: \n")
print("The Exceedances of GARCH(1,1) model is: "+str(exc_GARCH)+"\nThe Exceedances of empirical distribution method is: "+str(exc_emp)+"\nThe Exceedances of EWMA model is: "+str(exc_ewma))
print("The low bound of 5% confidence interval is "+str(CI_low)+"\nThe upper bound of 5% confidence interval is "+str(CI_high))
print("The GARCH and EWMA are good. The empirical distribution is too pessimistic. ")











