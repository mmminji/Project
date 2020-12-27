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
args.epoch = 2


# ====== Experiment Variable ====== #
name_var1 = 'lr'
name_var2 = 'n_layers'
list_var1 = [0.001]
list_var2 = [1]


trainset = solarDataset(file_loc, 'train', x_seq, y_seq)
valset = solarDataset(file_loc, 'val', x_seq, y_seq)
testset = solarTestDataset(file_loc, 'test', x_seq, y_seq)

partition = {'train': trainset, 'val':valset, 'test':testset}

for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1)
        setattr(args, name_var2, var2)
        print(args)
                
        setting, result = experiment(partition, deepcopy(args))

result