import pandas as pd
import numpy as np
import os
import glob
import random

import warnings
warnings.filterwarnings("ignore")

loc = 'C:/Users/a/OneDrive - 고려대학교/toyproject/태양열/data/'
train = pd.read_csv(loc + 'train/train.csv')
submission = pd.read_csv(loc + 'sample_submission.csv')

hot_time_0 = [0,1,2,3,4,5,6,7,8,18,19,20,21,22,23]
up = [6,7,8,9,10,11,12,13,14]

def add_feature(df):

    df['HOT_TIME'] = [0 if s in hot_time_0 else 1 for s in df['Hour']]
    df['UP'] = [1 if s in up else 0 for s in df['Hour']]

    df['winter'] = 0
    df['summer'] = 0

    for n in range(0, int(len(df)/48)-5):
        t1 = df[(n)*48: (n+6)*48]

        info = t1[t1.TARGET !=0].groupby(by='Hour').count().reset_index().iloc[0,:2]
        
        if (info.Hour == 8) | ((info.Hour ==7) & (info.Day >= 7)):
            df.iloc[n*48: (n+6)*48]['winter'] = 1
        elif ((info.Hour ==5) & (info.Day <= 11)):
            df.iloc[n*48: (n+6)*48]['summer'] = 1

    # df.loc[df[(df.winter ==1 ) &(df.summer==1)].index, ['winter','summer']] =0
    return df

aa = add_feature(train)
aa.to_csv('temp_2.csv')

def create_lag_feats(data, lags, cols):
    
    lag_cols = []
    temp = data.copy()
    for col in cols:
        for lag in lags:
            temp[col + '_lag_%s'%lag] = temp[col].shift(lag)
            temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
            temp['Target2'] = temp['TARGET'].shift(-96).fillna(method='ffill')
            lag_cols.append(col + '_lag_%s'%lag)

    return temp, lag_cols


def preprocess_data(data, target_lags=[48], weather_lags=[48], is_train=True):
    cols = data.columns.difference(['Day','Hour','Minute','TARGET'])
    temp = data.copy()

    if is_train==True:          
    
        temp, temp_lag_cols1 = create_lag_feats(temp, target_lags, ['TARGET'])
        temp, temp_lag_cols2 = create_lag_feats(temp, weather_lags, cols)
     
        return temp[['Hour'] + temp_lag_cols1 + temp_lag_cols2 + ['Target1', 'Target2']].dropna()

    elif is_train==False:    
        
        temp, temp_lag_cols1 = create_lag_feats(temp, target_lags, ['TARGET'])
        temp, temp_lag_cols2 = create_lag_feats(temp, weather_lags, cols)
                              
        return temp[['Hour'] + temp_lag_cols1 + temp_lag_cols2].dropna()


def add_diff(df, cols, lag_b, lag_a, is_train=True):

    if is_train == True:
        df_cols = df.columns.difference(['Hour', 'Target1', 'Target2']).tolist()

        diff_cols = []

        for col in cols:
            df[col + '_diff'] = df[col + '_lag_%s'%lag_a] - df[col + '_lag_%s'%lag_b]
            diff_cols.append(col + '_diff')

        return df[['Hour'] + df_cols + diff_cols + ['Target1', 'Target2']]

    elif is_train == False:
        df_cols = df.columns.difference(['Hour']).tolist()

        diff_cols = []

        for col in cols:
            df[col + '_diff'] = df[col + '_lag_%s'%lag_a] - df[col + '_lag_%s'%lag_b]
            diff_cols.append(col + '_diff')

        return df[['Hour'] + df_cols + diff_cols]



train = add_feature(train)
df_train = preprocess_data(train, target_lags=[0, 48, 96, 144, 192, 240, 288], weather_lags=[0, 48, 96, 144, 192, 240, 288], is_train=True)
# df_train = add_diff(df_train, ['DHI', 'DNI', 'RH', 'WS', 'T'], 48, 96, is_train=True)

df_test = []

for i in range(81):
    file_path = loc + 'test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)   
    temp = add_feature(temp)
    temp = preprocess_data(temp, target_lags=[0, 48, 96, 144, 192, 240, 288], weather_lags=[0, 48, 96, 144, 192, 240, 288], is_train=False).iloc[-48:]
    df_test.append(temp)

X_test = pd.concat(df_test)
# X_test = add_diff(X_test, ['DHI', 'DNI', 'RH', 'WS', 'T'], 48, 96, is_train=False)
# df_train[:48]
# X_test[:48]




import pandas as pd
import numpy as np
import os
import glob
import random
import math
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('../data/train/train.csv')
submission = pd.read_csv('../data/sample_submission.csv')

def prepro(df):
    hot_time_0 = [0,1,2,3,4,5,6,7,8,18,19,20,21,22,23]
    up = [6,7,8,9,10,11,12,13,14]

    df['HOT_TIME'] = [0 if s in hot_time_0 else 1 for s in df['Hour']]
    df['UP'] = [1 if s in up else 0 for s in df['Hour']]
    df['winter'] = 0
    df['summer'] = 0
    for n in range(0, int(len(df)/48)-5):
        t1 = df[(n)*48: (n+6)*48]

        info = t1[t1.TARGET !=0].groupby(by='Hour').count().reset_index().iloc[0,:2]
        
        if (info.Hour == 8) | ((info.Hour ==7) & (info.Day >= 7)):
            df.iloc[n*48: (n+6)*48]['winter'] = 1
        elif ((info.Hour ==5) & (info.Day <= 11)):
            df.iloc[n*48: (n+6)*48]['summer'] = 1

    df.loc[df[(df.winter ==1 ) &(df.summer==1)].index, ['winter','summer']] =0

    df['GHI'] = 0
    for d in tqdm(list(df.Day)):
        temp_Day = df[df['Day'] == d]
        temp_target = temp_Day[temp_Day['TARGET'] != 0]
        temp_len = int(len(temp_target))
        for i in range(0, temp_len, 2):
            n = temp_target.iloc[i].name
            df['GHI'][n] = df['DHI'][n] + df['DNI'][n] * abs(math.cos(i/temp_len * math.pi))
            df['GHI'][n+1] = df['DHI'][n+1] + df['DNI'][n+1] * abs(math.cos(i/temp_len * math.pi))
    return df

train = prepro(train)

train.to_csv('../data/train/train_prepro.csv')

train.loc[train[(train.winter ==1 ) &(train.summer==1)].index, ['winter','summer']] =0
