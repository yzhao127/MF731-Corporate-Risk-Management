# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:27:19 2019

@author: yyang
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm

# HW3
# Part I
# 2
K = 10
alpha = 0.99
gamma = 30
lam = 0.94
theta = 0.97
M = 30
N = 50000

data = pd.read_csv("C:\\Users\\yyang\\OneDrive\\Documents\\Documents\\BU\\Courses\\19Fall\\MF731\\HW\\HW3\\Prices.csv", header=0)
data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)

log_ret = np.log(data).diff().dropna()
n_ret = len(log_ret)

VaR_dt = []
es_dt = []
spec_dt = []

mkt_cap = np.array([196.94, 125.86, 131.57, 282.87])
mkt_cap_w = mkt_cap/np.sum(mkt_cap)

port_size = 1000000

mu_init = log_ret.iloc[0: M].mean()
cov_init = log_ret.iloc[0: M].cov()

for n in range(M, n_ret, 1): 
    cov_init = theta*cov_init+(1-theta)*((log_ret.iloc[n]-mu_init).values.reshape(4,1))*((log_ret.iloc[n]-mu_init).values.reshape(1,4))
    mu_init = lam*mu_init+(1-lam)*log_ret.iloc[n]

rnorm = np.random.multivariate_normal(mu_init, cov_init, N)
loss = -port_size*(np.exp(rnorm).dot(mkt_cap_w)-1) # One day loss
loss_sort = np.sort(loss)

var_dt = (K**0.5)*np.quantile(loss_sort, alpha)
es_dt = (K**0.5)*(1/(N*(1-alpha)))*(np.sum(loss_sort[math.ceil(N*alpha)-1: ])+(math.ceil(N*alpha)-N*alpha)*loss_sort[math.ceil(N*alpha)-1])
for i in range(N):
    spec_dt.append((np.exp(gamma*i/N)-np.exp(gamma*(i-1)/N))/(np.exp(gamma)-1))
    
spec = (K**0.5)*np.array(spec_dt).dot(loss_sort)

loss_K = []
for i in range(N): 
    log_ret_sim = np.zeros(shape=(K, 4))
    mu_t = mu_init
    cov_t = cov_init
    for j in range(K):
        log_ret_sim[j] = np.random.multivariate_normal(mu_t, cov_t, 1)
        cov_t = theta*cov_t+(1-theta)*((log_ret_sim[j]-mu_t).values.reshape(4,1))*((log_ret_sim[j]-mu_t).values.reshape(1,4))
        mu_t = lam*mu_t+(1-lam)*log_ret_sim[j]
    loss_K.append(-port_size*(np.prod(np.exp(log_ret_sim).dot(mkt_cap_w))-1))

loss_K_sort = np.sort(np.array(loss_K))

var_sim = np.quantile(loss_K_sort, alpha)
es_sim = (1/(N*(1-alpha)))*(np.sum(loss_K_sort[math.ceil(N*alpha)-1: ])+(math.ceil(N*alpha)-N*alpha)*loss_K_sort[math.ceil(N*alpha)-1])
spec_sim = np.array(spec_dt).dot(loss_K_sort)


# Part II
# 1
alpha = 0.99
M = 50
lam = 0.94
theta= 0.96
port_size = 15000000

data = pd.read_csv("C:\\Users\\yyang\\OneDrive\\Documents\\Documents\\BU\\Courses\\19Fall\\MF731\\HW\\HW3\\Five_Stock_Prices.csv", header=0)
data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)
dates = data.index

log_ret = np.log(data).diff().dropna()
N = len(log_ret)

dollar_pos = np.array([[port_size*0.2] * 5] * N)

var_data = np.zeros(shape=(N-M+1, 1))
es_data = np.zeros(shape=(N-M+1, 1))
mar_var_data = np.zeros(shape=(N-M+1, 5))
mar_es_data = np.zeros(shape=(N-M+1, 5))
comp_var_data = np.zeros(shape=(N-M+1, 5))
comp_es_data = np.zeros(shape=(N-M+1, 5))
rel_comp_var_data = np.zeros(shape=(N-M+1, 5))
rel_comp_es_data = np.zeros(shape=(N-M+1, 5))
pct_variance_data = np.zeros(shape=(N-M+1, 5))
date_vals = []

mu = log_ret.iloc[0:M].mean()
cov = log_ret.iloc[0:M].cov()

for i in range(M, N, 1):
    date_vals.append(dates[i+1])
    cov = theta*cov + (1-theta)*((log_ret.iloc[i]-mu).values.reshape(5,1))*((log_ret.iloc[i]-mu).values.reshape(1,5))
    mu = lam*mu+(1-lam)*log_ret.iloc[i]
    var_data[i+1-M] = -dollar_pos[i].dot(mu) + ((dollar_pos[i].dot(cov.dot(dollar_pos[i])))**0.5)*norm.ppf(alpha)
    es_data[i+1-M] = -dollar_pos[i].dot(mu) + ((dollar_pos[i].dot(cov.dot(dollar_pos[i])))**0.5)*(1/(1-alpha))*norm.pdf(norm.ppf(alpha))
    mar_var_data[i+1-M] = -mu + norm.ppf(alpha)*(np.transpose(cov.dot(dollar_pos[i])))/((dollar_pos[i].dot(np.transpose(cov.dot(dollar_pos[i]))))**0.5)
    mar_es_data[i+1-M] = -mu + (1/(1-alpha))*norm.pdf(norm.ppf(alpha))*(np.transpose(cov.dot(dollar_pos[i])))/((dollar_pos[i].dot(np.transpose(cov.dot(dollar_pos[i]))))**0.5)
    comp_var_data[i+1-M] = mar_var_data[i+1-M] * dollar_pos[i]
    comp_es_data[i+1-M] = mar_es_data[i+1-M] * dollar_pos[i]
    rel_comp_var_data[i+1-M] = 100*comp_var_data[i+1-M]/var_data[i+1-M]
    rel_comp_es_data[i+1-M] = 100*comp_es_data[i+1-M]/es_data[i+1-M]
    pct_variance_data[i+1-M] = 100*(np.transpose(cov.dot(dollar_pos[i]))*dollar_pos[i]/(dollar_pos[i].dot(np.transpose(cov.dot(dollar_pos[i])))))

df_rel_comp_var_data = pd.DataFrame(rel_comp_var_data[1: ], index=date_vals, columns=data.columns)
df_rel_comp_var_data.plot()

df_rel_comp_es_data = pd.DataFrame(rel_comp_es_data[1: ], index=date_vals, columns=data.columns)
df_rel_comp_es_data.plot()

df_pct_variance_data = pd.DataFrame(pct_variance_data[1: ], index=date_vals, columns=data.columns)
df_pct_variance_data.plot()
