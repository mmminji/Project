#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.formula.api import ols
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

import os

import pandas as pd
import pandas_datareader.data as pdr

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-whitegrid')

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
from itertools import repeat, chain
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error


# In[2]:


def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[3]:


i=0
seasonal = pd.read_csv('C:/Users/a/OneDrive - 고려대학교/toyproject/태양열/data/test/{}.csv'.format(i))
seasonal


# In[10]:


df = pd.DataFrame()

for i in range(82):
    seasonal = pd.read_csv('C:/Users/a/OneDrive - 고려대학교/toyproject/태양열/data/test/{}.csv'.format(i))
    auto_arima_model = auto_arima(seasonal.TARGET, 
                             start_p=0, start_q=0,
                             max_p=2, max_q=2, m =48, seasonal=True,
                             d=1, D=1,
                             max_P=1, max_Q=1,
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=False)
    prediction = auto_arima_model.predict(96, return_conf_int=True)
    pred_value = prediction[0]
    pred_up = prediction[1][:,0]
    pred_lb = prediction[1][:,1]
    dff = pd.DataFrame([pred_value, pred_up, pred_lb])
    df = pd.concat([df,dff])
    


# In[11]:


df


# In[48]:


data = pd.DataFrame()

for i in range(81):
    a = df.iloc[i*3]
    data = pd.concat([data, a])


# In[49]:


data = data.rename(columns = {0: 0.5})
data[0.1] = data[0.5]*0.6
data[0.2] = data[0.5]*0.7
data[0.3] = data[0.5]*0.8
data[0.4] = data[0.5]*0.9
data[0.6] = data[0.5]*1.1
data[0.7] = data[0.5]*1.2
data[0.8] = data[0.5]*1.3
data[0.9] = data[0.5]*1.4
data = data[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
data = data.round(2)
data


# In[50]:


data.to_csv('arima.csv')


# In[ ]:




