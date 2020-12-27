import os
import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

# file_loc = 'C:/Users/a/OneDrive - 고려대학교/toyproject/태양열/data/'
# x_seq = 336 
# y_seq = 96   
# pd.read_csv(file_loc + 'train/train' + '.csv').TARGET.max() = 99.9139


# data = pd.read_csv(file_loc + '0.csv')
# for i in range(y_seq):
#     data = data.append(pd.DataFrame([(0,0,0,0,0,0,0,0,0)],
#                                              columns = ['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'],
#                                              ), ignore_index=True)
# data[0+x_seq:0+x_seq+y_seq].values


class solarDataset(Dataset):

    def __init__(self, file_loc, tvt, x_seq, y_seq):

        self.file_loc = file_loc
        self.tvt = tvt  # train, val
        self.x_seq = x_seq
        self.y_seq = y_seq

        if self.tvt == 'train':
            self.data = pd.read_csv(self.file_loc + 'train/train.csv')[:42000]
        elif self.tvt == 'val': 
            self.data = pd.read_csv(self.file_loc + 'train/train.csv')[42000:]
        else:
            pass

        # scaler = MinMaxScaler()
        self.data['TARGET'] = self.data['TARGET'] / 100

        
    def __len__(self):
        return len(self.data) - (self.x_seq + self.y_seq) + 1

    def __getitem__(self, idx):        
        # self.data = self.data.apply(lambda x : np.log(x+1) - np.log(x[self.x_seq-1]+1))
        X = self.data.iloc[idx:idx+self.x_seq].values
        y = self.data.iloc[idx+self.x_seq:idx+self.x_seq+self.y_seq].values
        return X, y


class solarTestDataset(Dataset):
    def __init__(self, file_loc, tvt, x_seq, y_seq):
    
        self.file_loc = file_loc
        self.tvt = tvt  # test
        self.x_seq = x_seq
        self.y_seq = y_seq

        if self.tvt == 'test':
            self.data = pd.DataFrame()
            for f in glob.glob(self.file_loc + 'test/*.csv'):
                df = pd.read_csv(f)
                self.data = self.data.append(df, ignore_index=True)
        else:
            pass

        # scaler = MinMaxScaler()
        self.data['TARGET'] = self.data['TARGET'] / 100

        
    def __len__(self):
        return int(len(self.data) / self.x_seq)

    def __getitem__(self, idx):        
        # self.data = self.data.apply(lambda x : np.log(x+1) - np.log(x[self.x_seq-1]+1))
        X = self.data.iloc[idx*self.x_seq:(idx+1)*self.x_seq].values
        return X

# dataset = solarDataset(file_loc, 'train', x_seq, y_seq)
# dataloader = DataLoader(dataset, 100, drop_last=True, shuffle=False)

# trainset = solarTestDataset(file_loc, 'test', x_seq, y_seq)
# trainloader = DataLoader(trainset, 
#                              batch_size=1, 
#                              shuffle=True, drop_last=True)

# len(trainloader)
# for X, y in trainloader:
#     print(X.shape, y.shape)
#     X = X[:, :, 8:9].transpose(0, 1).float().to('cuda')   # np.swapaxes와 같은 역할, batch_size와 seq_len의 위치를 바꿔줌(위에가 원하는 형태)
#     y_true = y[:, :, 8].float().to('cuda') 
#     print(X.shape, y_true.shape) 
#     break
