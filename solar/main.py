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
args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
args.batch_size = 81
# args.x_frames = 10
args.y_frames = 96
file_loc = 'C:/Users/a/OneDrive - 고려대학교/toyproject/태양열/data/'
x_seq = 336 
y_seq = 96   

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
args.epoch = 30
args.quantile = 0.9

# ====== Experiment Variable ====== #
name_var1 = 'n_layers'
name_var2 = 'quantile'
list_var1 = [2]
list_var2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]



trainset = solarDataset(file_loc, 'train', x_seq, y_seq)
valset = solarDataset(file_loc, 'val', x_seq, y_seq)
testset = solarTestDataset(file_loc, 'test', x_seq, y_seq)

partition = {'train': trainset, 'val':valset, 'test':testset}

df = pd.DataFrame()
for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1)
        setattr(args, name_var2, var2)
        print(args)
                
        setting, result = experiment(partition, deepcopy(args))
        df[args.quantile] = result['y_pred'].view(-1).to('cpu')
    df = df * 100
df.to_csv('sub_100_e30_p_layer2.csv', index=False)

# sub = pd.DataFrame(result['y_pred'].view(-1).to('cpu'))
# df = pd.read_csv('sub_w100_e10_p_0.1.csv')
# df[args.quantile] = result['y_pred'].view(-1).to('cpu')
# sub['1'] = sub[0]*0.1
# sub['2'] = sub[0]*0.2
# sub['3'] = sub[0]*0.3
# sub['4'] = sub[0]*0.4
# sub['5'] = sub[0]*0.5
# sub['6'] = sub[0]*0.6
# sub['7'] = sub[0]*0.7
# sub['8'] = sub[0]*0.8
# sub['9'] = sub[0]*0.9
# df = df * 100
# df.to_csv('sub_w100_e10_p_0.1.csv', index=False)
# df

