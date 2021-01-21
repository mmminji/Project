import pandas as pd
import numpy as np
import os
import glob
import random
import math

import warnings
warnings.filterwarnings("ignore")

loc = 'C:/Users/a/OneDrive - 고려대학교/toyproject/태양열/data_ghi/'
train = pd.read_csv(loc + 'train/train_prepro.csv', index_col = 0)
submission = pd.read_csv(loc + 'sample_submission.csv')


def hot_time1(df):
    df['HOT_TIME1'] = 0
    df['HOT_TIME2'] = 0
    for i in list(set(df.Day)):
        for j in list(df[df['Day'] == i].index):
            if df['TARGET'][j] != 0:
                break
        df['HOT_TIME1'][j-1] = 1
        df['HOT_TIME1'][j] = 1
        df['HOT_TIME1'][j+1] = 1

        for k in list(df[df['Day'] == i].index)[j-48*i:]:
            if df['TARGET'][k] == 0:
                break
        df['HOT_TIME2'][k-2] = 1
        df['HOT_TIME2'][k-1] = 1
        df['HOT_TIME2'][k] = 1

    for i in df[df['TARGET'] == 0].index:
        df['HOT_TIME1'][i] = 0
        df['HOT_TIME2'][i] = 0

    return df

def hot_time2(df):
    df['HOT_TIME'] = 1
    for i in list(set(df.Day)):
        for j in list(df[df['Day'] == i].index):
            if df['TARGET'][j] != 0:
                break
        df['HOT_TIME'][j-1] = 0
        df['HOT_TIME'][j] = 0
        df['HOT_TIME'][j+1] = 0

        for k in list(df[df['Day'] == i].index)[j-48*i:]:
            if df['TARGET'][k] == 0:
                break
        df['HOT_TIME'][k-2] = 0
        df['HOT_TIME'][k-1] = 0
        df['HOT_TIME'][k] = 0

    for i in df[df['TARGET'] == 0].index:
        df['HOT_TIME'][i] = 0

    return df

def add_daylight(train):
    train['daylight_Hours'] = 0

    for i in range(train.Day.nunique()):
        hours = train[(train['Day']==i) & (train['DHI']!=0)]['daylight_Hours'].groupby(train['Day']).count()
        # train[train['Day']==i]['daylight_Hours'] = hours.values[0]
        train[i*48:(i+1)*48]['daylight_Hours'] = hours.values[0]

    return train

def add_time(train):
    train['Temp'] = [0.5 if s == 30 else 0 for s in train['Minute']]
    train['New_Hour'] = train['Temp'] + train['Hour']
    train['Hour'] = train['New_Hour']
    train = train.drop(['Temp', 'New_Hour'], axis=1)
    return train

def add_season(df):
    df['dhi_win'] = 0
    df['dhi_sum'] = 0
    df['dhi_s_f'] = 0
    for n in range(0, math.ceil(len(df)/48/6)):
        t1 = df[(6*n)*48: ((6*n)+6)*48]

        if t1.DHI.max() < 275.5:
            df[(6*n)*48: ((6*n)+6)*48]['dhi_win'] = 1
        elif (t1.DHI.max() >= 275.5) & (t1.DHI.max() < 488.0):
            df[(6*n)*48: ((6*n)+6)*48]['dhi_s_f'] = 1
        elif t1.DHI.max() > 488.0:
            df[(6*n)*48: ((6*n)+6)*48]['dhi_sum'] = 1
    return df

def humid_cloud(train):
    train['high_humid'] = 0
    train.loc[train[train.RH > 77].index, 'high_humid'] = 1
    train['cloud'] = 0
    train.loc[train[(train['T'] < 0) & (train.RH > 77)].index, 'cloud'] = 1
    return train

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
            df[col + '_diff_' + str(lag_a)] = df[col + '_lag_%s'%lag_a] - df[col + '_lag_%s'%lag_b]
            diff_cols.append(col + '_diff_' + str(lag_a))

        return df[['Hour'] + df_cols + diff_cols + ['Target1', 'Target2']]

    elif is_train == False:
        df_cols = df.columns.difference(['Hour']).tolist()

        diff_cols = []

        for col in cols:
            df[col + '_diff_' + str(lag_a)] = df[col + '_lag_%s'%lag_a] - df[col + '_lag_%s'%lag_b]
            diff_cols.append(col + '_diff_' + str(lag_a))

        return df[['Hour'] + df_cols + diff_cols]



from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.preprocessing import StandardScaler
stand_scaler = StandardScaler()

# train[train.columns.difference(['TARGET', 'Day'])] = stand_scaler.fit_transform(train[train.columns.difference(['TARGET', 'Day'])])
train = train.drop(['winter', 'summer'], axis=1)
train = add_season(train)
train = add_daylight(train)
train = hot_time1(train)
train = hot_time2(train)
train = humid_cloud(train)
df_train = preprocess_data(train, target_lags=[0, 48, 96, 144, 192, 240, 288], weather_lags=[0, 48, 96, 144, 192, 240, 288], is_train=True)
df_train = add_diff(df_train, ['DHI', 'DNI', 'RH', 'WS', 'T'], 0, 48, is_train=True)
df_train = add_diff(df_train, ['DHI', 'DNI', 'RH', 'WS', 'T'], 48, 96, is_train=True)


df_test = []

for i in range(81):
    file_path = loc + 'test/' + str(i) + '_prepro.csv'
    temp = pd.read_csv(file_path, index_col = 0)
    # temp[temp.columns.difference(['TARGET', 'Day'])] = stand_scaler.transform(temp[temp.columns.difference(['TARGET', 'Day'])])
    temp = temp.drop(['winter', 'summer'], axis=1)
    temp = add_season(temp)
    temp = add_daylight(temp)
    temp = hot_time1(temp)
    temp = hot_time2(temp)
    temp = humid_cloud(temp)
    temp = preprocess_data(temp, target_lags=[0, 48, 96, 144, 192, 240, 288], weather_lags=[0, 48, 96, 144, 192, 240, 288], is_train=False).iloc[-48:]
    df_test.append(temp)


X_test = pd.concat(df_test)
X_test = add_diff(X_test, ['DHI', 'DNI', 'RH', 'WS', 'T'], 0, 48, is_train=False)
X_test = add_diff(X_test, ['DHI', 'DNI', 'RH', 'WS', 'T'], 48, 96, is_train=False)

# df_train[:48]
# df_train
# X_test[:48]
# df_train.to_csv('dftrain.csv', index=False)