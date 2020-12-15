import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import time
import torch
from tqdm.notebook import tqdm
from torchvision import transforms
from torch import nn
# import galaxy_Dataset as GD
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
# import galaxy_model
from model_resnet import *
import neptune
from dataloader import *

#########################
# configs
folder_loc = 'C:/Users/a/OneDrive - 고려대학교/toyproject/딥러닝'
batch_sizes = 256
learning_rate = 0.005
patience = 10
trMaxEpoch = 100
# imgtransResize = (224, 224) #256
# imgtransCrop = 224
checkpoint = None
model_name = 'resnet152'
pretrain_check = False
pathModel = folder_loc+'/checkpoint/'+ model_name +'/'  #저장할 위치
#########################

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
# transformList.append(transforms.Resize(imgtransResize))
# transformList.append(transforms.RandomResizedCrop(imgtransCrop))
# transformList.append(transforms.ToPILImage())
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.RandomRotation(5))
transformList.append(transforms.ColorJitter(contrast=[0.75, 1.25]))
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformTrain = transforms.Compose(transformList)

transformList = []
# transformList.append(transforms.ToPILImage())
# transformList.append(transforms.Resize(imgtransResize))
# transformList.append(transforms.RandomResizedCrop(imgtransCrop))
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformVal = transforms.Compose(transformList)

# train, validation 데이터 셋 정의
train_path = folder_loc +'/data/train_galaxy'

indices = list(range(len(GalD(train_path))))
train_idx, valid_idx = train_test_split(indices,test_size=0.25)
dataset = GalD(train_path, transform=None)
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, valid_idx)

train_dataset = MapDataset(train_dataset, transformTrain)
val_dataset = MapDataset(val_dataset, transformVal)

DT_train = torch.utils.data.DataLoader(train_dataset, batch_size = batch_sizes, shuffle =True)
DT_val = torch.utils.data.DataLoader(val_dataset, batch_size = batch_sizes, shuffle =True)


# 모델 정의
if model_name == 'resnet18':
    model = resnet18(pretrained=pretrain_check).cuda()
elif model_name == 'resnet34':
    model = resnet34(pretrained=pretrain_check).cuda()
elif model_name == 'resnet50':
    model = resnet50(pretrained=pretrain_check).cuda()
elif model_name == 'resnet101':
    model = resnet101(pretrained=pretrain_check).cuda()
elif model_name == 'resnet152':
    model = resnet152(pretrained=pretrain_check).cuda()
elif model_name == 'resnext50_32x4d':
    model = resnext50_32x4d(pretrained=pretrain_check).cuda()
elif model_name == 'resnext101_32x8d':
    model = resnext101_32x8d(pretrained=pretrain_check).cuda()
elif model_name == 'wide_resnet50_2':
    model = wide_resnet50_2(pretrained=pretrain_check).cuda()
elif model_name == 'wide_resnet101_2':
    model = wide_resnet101_2(pretrained=pretrain_check).cuda()

model = torch.nn.DataParallel(model).cuda()

# optimizer, scheduler, criterion 정의
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, factor=0.01, patience=patience, mode='min')
criterion = nn.CrossEntropyLoss()
lossMIN = 100000

neptune.init(project_qualified_name='kbh/gggg',
api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMWUzMmM0NjgtOTgxZC00NjQ0LWEwYWYtMDhkZWE3NGUzOGM4In0=')

def NeptuneLog():
    neptune.log_metric('batch_size',batch_sizes) 
    neptune.log_metric('learning_rate',learning_rate)
    neptune.log_text('pre-trained', str(pretrain_check))
    neptune.log_text('model',model_name)

neptune.create_experiment()
NeptuneLog()


if checkpoint != None:
    modelCheckpoint = torch.load(checkpoint)
    model.load_state_dict(modelCheckpoint['state_dict'], strict=False)
    optimizer.load_state_dict(modelCheckpoint['optimizer'])
    print('####### Checkpoint Restored #####')



for epochID in range(0, trMaxEpoch):
    print(f'Epoch : {epochID+1} / {trMaxEpoch}')
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    print('current lr : {:.7f}'.format(optimizer.param_groups[0]['lr']))
    #train
    lossTrain = 0
    lossTraNorm = 0
    model.train()
    for images, labels in DT_train:
        images, labels = images.cuda(), labels.cuda()  ##
        output = model(images)
        labels = torch.reshape(labels, (len(labels),))
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward() ##
        optimizer.step()
        
        lossTrain += loss
        lossTraNorm += 1 
    T_loss = lossTrain/lossTraNorm
    #val
    with torch.no_grad():
        model.eval()
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0
        for images, labels in DT_val:
            images, labels = images.cuda(), labels.cuda()  ##
            output = model(images)
            labels = torch.reshape(labels, (len(labels),))
            loss = criterion(output, labels)
            losstensorMean += loss
            lossVal += loss
            lossValNorm += 1
        
       

        outLoss = lossVal / lossValNorm
    

    # neptune.log_metric('Train loss', T_loss)    
    neptune.log_metric('Val loss', lossVal)

    print('Train Loss : ', T_loss.item() ,'\t Val Loss : ', outLoss.item())
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampEND = timestampDate + '-' + timestampTime

    scheduler.step(lossVal) ##

    if lossVal < lossMIN:
        lossMIN = lossVal
        torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                    'optimizer': optimizer.state_dict()}, f'{pathModel}_{epochID+1}.pt')  ##

        print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
        neptune.log_metric('min Val loss', lossMIN)
        neptune.log_metric('save Epoch', epochID+1)
    else:
        print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
neptune.stop()

