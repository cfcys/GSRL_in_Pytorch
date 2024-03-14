# -*- coding: utf-8 -*-
import os
import argparse
# from keras.callbacks import EarlyStopping
# from keras import losses
from tqdm import tqdm
# from tensorflow.keras.utils import to_categorical
# from keras.optimizers import Adadelta
from sklearn.metrics.pairwise import paired_distances as dist
# from Hyper import imgDraw, listClassification, resnet99_avg_recon
from Hyper import imgDraw
import libmr
import numpy as np
import rscls
import glob
from scipy import io
from copy import deepcopy
import time
import torch.nn.functional as F

# new
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from keras.utils import to_categorical
from Hypertorch2 import ResNet99
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from utils import to_categorical
from torch.optim.lr_scheduler import LambdaLR   # 引入学习率调度的工具LambdaLR
import ipdb
# 定义学习率调整函数


class CustomDataset(Dataset):
    def __init__(self, data, lidar_data, labels):
        self.data = data
        self.lidar_data = lidar_data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        lidar_sample = self.lidar_data[idx]
        label = self.labels[idx]
        return data_sample, lidar_sample, label


# 设置args
os.environ['CUDA_VISIBLE_DEVICES']='1'
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--numTrain', type=int, default=20)
parser.add_argument('--dataset', type=str, default='data/Trento/trento_im.npy') 
# parser.add_argument('--dataset', type=str, default='data/Houston/houston_im.npy') 
parser.add_argument('--gt', type=str, default='data/Trento/trento_raw_gt.npy')
# parser.add_argument('--gt', type=str, default='data/Houston/houston_raw_gt.npy')
parser.add_argument('--num_epochs1', type=int, default=270) 
parser.add_argument('--num_epochs2', type=int, default=230) 
parser.add_argument('--batch_size', type=int, default=32) 
parser.add_argument('--output', type=str, default='output/')
args = parser.parse_args()
# early_stopping = EarlyStopping(monitor='loss', patience=1000)
key = args.dataset.split('/')[-1].split('_')[0]
spath = args.output + key + '_' + str(args.numTrain) + '/'
os.makedirs(spath, exist_ok=True)


def lr_lambda(epoch):
    return 1.0 if epoch < args.num_epochs1 else 0.1

# 加载数据，并分配好雷达数据和高光谱数据
hsi = np.load(args.dataset).astype('float32')
gt = np.load(args.gt).astype('int')
num_classes = np.max(gt)
row, col, layers = hsi.shape

# 搞一下训练的数据集
c1 = rscls.rscls(hsi, gt, cls=num_classes)
c1.padding(9)

x1_train, y1_train = c1.train_sample(args.numTrain) 

x1_train, y1_train = rscls.make_sample(x1_train, y1_train) 
x1_lidar_train = x1_train[:, :, :, -1]
x1_lidar_train = x1_lidar_train[:, :, :, np.newaxis]
x1_train = x1_train[:, :, :, :-1]
y1_train = to_categorical(y1_train, num_classes)

# 查看当前的输入内容的形状
print("原始读入数据的输入：")
print(x1_train.shape)
print(x1_lidar_train.shape)
x1_train = x1_train.transpose((0,3,1,2))
x1_lidar_train = x1_lidar_train.transpose((0,3,1,2))
print("经过维度调整之后：")
print(x1_train.shape)
print(x1_lidar_train.shape)

# 写训练所用的Dataset类和data_loader
train_dataset = CustomDataset(torch.tensor(x1_train), torch.tensor(x1_lidar_train),torch.tensor(y1_train))

# 写一个data_loader
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# 设置其他训练时候所用的参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model = ResNet99(layers - 1, 1, 9, num_classes).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.L1Loss()

scheduler = LambdaLR(optimizer, lr_lambda)

np.set_printoptions(threshold=np.inf)


start_time = time.time()
# for epoch in tqdm(range(args.num_epochs1 + args.num_epochs2)):
for epoch in range(args.num_epochs1 + args.num_epochs2):
    model.train()                                # 用于将模型设置为训练模式
    for x, x_lidar, y in train_data_loader:
        x = torch.FloatTensor(x).to(device)  # 确保x是float
        x_lidar = torch.FloatTensor(x_lidar).to(device)  # 确保x_lidar是float
        
        y = y.to(device).float()

        optimizer.zero_grad()

        output1, output2 = model(x,x_lidar)  # 假设你的模型接受一个输入
        
        loss1 = criterion1(output1,torch.argmax(y,dim=1))
        loss2 = criterion2(output2,x)
        
        loss = 0.5 * loss1 + 0.5 * loss2
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # 在每个epoch结束时改变学习率
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

end_time = time.time()
print('Training time:', end_time - start_time)

model.eval()  

time3 = int(time.time())
print('Start predicting...')

model.eval()   # 将模型设置为评估模式，主要是禁用Batchnorm和Dropout
pre_all = []
pre_loss = []
with torch.no_grad():  # 确保在预测过程中不计算梯度
    for r in tqdm(range(row)):
        # 假设 c1.all_sample_row(r) 已经准备好了适用于 PyTorch 的数据
        row_samples = c1.all_sample_row(r)  # 需要根据实际情况调整
        
        row_samples_tensor = torch.tensor(row_samples, dtype=torch.float32)

        x_samples = row_samples_tensor[:, :, :, :-1]
        lidar_samples = row_samples_tensor[:, :, :, -1].unsqueeze(-1)
        x_samples = x_samples.permute(0,3,1,2).to(device)
        lidar_samples = lidar_samples.permute(0,3,1,2).to(device)
        pre_row, recons = model(x_samples, lidar_samples)
        
        pre_all.append(pre_row.cpu().numpy())
        recons_loss = dist(recons.cpu().reshape(col, -1), x_samples.cpu().reshape(col, -1))
        pre_loss.append(recons_loss)

pre_all = np.array(pre_all).astype('float64')
pre_loss = np.array(pre_loss).reshape(-1).astype('float64')

    
    # 训练集上的重建损失
_, recons_train = model(torch.FloatTensor(x1_train).to(device), torch.FloatTensor(x1_lidar_train).to(device))
train_loss = dist(recons_train.reshape(recons_train.shape[0], -1).cpu().detach().numpy(), 
                  x1_train.reshape(x1_train.shape[0], -1)) 



time4 = int(time.time())
print('predict time:',time4-time3)

print('Start caculating open-set...')
mr = libmr.MR()
mr.fit_high(train_loss, 20)
wscore = mr.w_score_vector(pre_loss.astype(np.float64))
mask = wscore > 0.5 
mask = mask.reshape(row, col)
unknown = gt.max() + 1

# for close set
pre_closed = np.argmax(pre_all, axis=-1) + 1     
imgDraw(pre_closed, spath + key + '_closed', path='./', show=False)

# for open set
pre_gsrl = deepcopy(pre_closed)
pre_gsrl[mask == 1] = unknown 
gt_new = deepcopy(gt)

gt2file = glob.glob('data/Trento/' + key + '*gt*[0-9].npy')[0]
# gt2file = glob.glob('data/Houston/' + key + '*gt*[0-9].npy')[0]

gt2 = np.load(gt2file)
gt_new[np.logical_and(gt_new == 0, gt2 != 0)] = unknown
cfm = rscls.gtcfm(pre_gsrl, gt_new, unknown)

pre_to_draw = deepcopy(pre_gsrl)
pre_to_draw[pre_to_draw == unknown] = 0
imgDraw(pre_to_draw, spath + key + '_gsrl', path='./', show=False)




# c1 = rscls.rscls(hsi, gt, cls=numClass)
# c1.padding(9)
# x1_train, y1_train = c1.train_sample(args.numTrain) 
# x1_train, y1_train = rscls.make_sample(x1_train, y1_train) 
# x1_lidar_train = x1_train[:, :, :, -1]
# x1_lidar_train = x1_lidar_train[:, :, :, np.newaxis]
# x1_train = x1_train[:, :, :, :-1]
# y1_train = to_categorical(y1_train, numClass) 

# print('Start training...')
# time2 = int(time.time())
# model, _ = resnet99_avg_recon(layers - 1, 1, 9, numClass, l=1,latent_dim=64)
# model.compile(loss=['categorical_crossentropy', losses.mean_absolute_error], optimizer=Adadelta(lr=1.0),
#               metrics=['accuracy'], loss_weights=[0.5, 0.5])
# model.fit([x1_train, x1_lidar_train], [y1_train, x1_train], batch_size=args.batch_size,
#           epochs=270, verbose=1, shuffle=True, callbacks=[early_stopping])
# model.compile(loss=['categorical_crossentropy', losses.mean_absolute_error], optimizer=Adadelta(lr=0.1),
#               metrics=['accuracy'], loss_weights=[0.5, 0.5])
# model.fit([x1_train, x1_lidar_train], [y1_train, x1_train], batch_size=args.batch_size,
#           epochs=230, verbose=1, shuffle=True, callbacks=[early_stopping])
# model.save(spath + key + '_model')   # 保存训练好的模型，用于下面的预测
# # model.save(spath)
# time3 = int(time.time())
# print('training time:',time3-time2)


# print('Start predicting...')
# pre_all = []
# pre_loss = []
# for r in tqdm(range(row)):
#     row_samples = c1.all_sample_row(r)
#     pre_row, recons = model.predict([row_samples[:, :, :, :-1], (row_samples[:, :, :, -1])[:, :, :, np.newaxis]])
#     pre_all.append(pre_row)
#     recons_loss = dist(recons.reshape(col, -1), row_samples[:, :, :, :-1].reshape(col, -1))
#     pre_loss.append(recons_loss)
# pre_all = np.array(pre_all).astype('float64')
# pre_loss = np.array(pre_loss).reshape(-1).astype('float64')
# recons_train = model.predict([x1_train, x1_lidar_train])[1]
# train_loss = dist(recons_train.reshape(recons_train.shape[0], -1), x1_train.reshape(x1_train.shape[0], -1))

# time4 = int(time.time())
# print('predict time:',time4-time3)

# print('Start caculating open-set...')
# mr = libmr.MR()
# mr.fit_high(train_loss, 20)
# wscore = mr.w_score_vector(pre_loss)
# mask = wscore > 0.5 
# mask = mask.reshape(row, col)
# unknown = gt.max() + 1

# # for close set
# pre_closed = np.argmax(pre_all, axis=-1) + 1 
# imgDraw(pre_closed, spath + key + '_closed', path='./', show=False)

# # for open set
# pre_gsrl = deepcopy(pre_closed)
# pre_gsrl[mask == 1] = unknown 
# gt_new = deepcopy(gt)

# gt2file = glob.glob('data/Trento/' + key + '*gt*[0-9].npy')[0]
# # gt2file = glob.glob('data/Houston/' + key + '*gt*[0-9].npy')[0]

# gt2 = np.load(gt2file)
# gt_new[np.logical_and(gt_new == 0, gt2 != 0)] = unknown
# cfm = rscls.gtcfm(pre_gsrl, gt_new, unknown)

# pre_to_draw = deepcopy(pre_gsrl)
# pre_to_draw[pre_to_draw == unknown] = 0
# imgDraw(pre_to_draw, spath + key + '_gsrl', path='./', show=False)
