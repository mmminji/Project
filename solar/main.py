from dataloader import *
import numpy as np
import torch
import argparse
from copy import deepcopy
import torch.optim as optim
from train import *
from validation import *
from test import *
from LSTM import *
from GRU import *
from exp import *

# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
# args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
args.batch_size = 81
# args.x_frames = 10
args.y_frames = 2  # y_seq와 동일
file_loc = 'C:/Users/a/OneDrive - 고려대학교/toyproject/태양열/data/'
x_seq = 7 
y_seq = 2   

# ====== Model Capacity ===== #
args.input_dim = 1
args.hid_dim = 50
args.n_layers = 2

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.0
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'RMSprop' #'RMSprop' #SGD, RMSprop, ADAM...
args.lr = 0.0001
args.epoch = 50
# args.quantile = 0.9

# ====== Experiment Variable ====== #
name_var1 = 'n_layers'
# name_var2 = 'quantile'
name_var2 = 'lr'
list_var1 = [2]
# list_var2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
list_var2 = [0.0001]

sub_concat = pd.DataFrame()
for hour in range(24):
    for minute in [0, 30]:
        print('#==============', hour, minute, '==============#')

        trainset = solarDataset(file_loc, 'train', x_seq, y_seq, hour, minute)
        valset = solarDataset(file_loc, 'val', x_seq, y_seq, hour, minute)
        testset = solarTestDataset(file_loc, 'test', x_seq, y_seq, hour, minute)

        partition = {'train': trainset, 'val':valset, 'test':testset}

        # ======= 범위 존재 ====== #
        # sub = pd.DataFrame()
        # for var1 in list_var1:
        #     for var2 in list_var2:
        #         setattr(args, name_var1, var1)
        #         setattr(args, name_var2, var2)
        #         print(args)
                        
        #         setting, result = experiment(partition, deepcopy(args))
        #         sub[args.quantile] = result['y_pred'].view(-1).to('cpu')
        #     sub = sub * 100
        # # sub.to_csv('submission/sub_100_e30_m_layer2.csv', index=False)


        # ======== 범위 존재안함 ========== #
        print(args)
        setting, result = experiment(partition, deepcopy(args))
        sub = pd.DataFrame(result['y_pred'].view(-1).to('cpu'))

        sub['1'] = sub[0]*0.6
        sub['2'] = sub[0]*0.7
        sub['3'] = sub[0]*0.8
        sub['4'] = sub[0]*0.9
        sub['5'] = sub[0]*1
        sub['6'] = sub[0]*1.1
        sub['7'] = sub[0]*1.2
        sub['8'] = sub[0]*1.3
        sub['9'] = sub[0]*1.4
        sub = sub * 100
        sub_concat = pd.concat([sub_concat, sub])
        sub_concat = sub_concat.sort_index()
        # sub.to_csv('submission/sub_100_e30_m_0.5.csv', index=False)

# ======== 후처리 ========== #
import pandas as pd
from tqdm import trange

sub = sub_concat.copy()
file_id = pd.read_csv('C:/Users/a/OneDrive - 고려대학교/toyproject/태양열/data/sample_submission.csv')
sub.index = file_id['id']
sub = sub.reset_index()

result = pd.read_csv('temp.csv')

col_list = sub.columns.difference(['id'])

for i in trange(len(result)):
    index = sub[sub['id'] == result['file'][i]].index.values[0]
    for col in col_list:
        sub[col][index] = 0

# sub
sub = sub.drop([0], axis=1)
sub.columns = ['id', 'q_0.1', 'q_0.2', 'q_0.3', 'q_0.4', 'q_0.5', 'q_0.6', 'q_0.7',
       'q_0.8', 'q_0.9']

sub.to_csv('submission/sub_m_l2_h50_in1_e50_m48.csv', index=False)

