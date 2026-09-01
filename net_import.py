"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn
import numpy as np
#import transforms
import os
import math
import argparse
import torch.utils.data as Data
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
#import pytorch_lightning as pl
#from torchvision import transforms

import matplotlib.pyplot as plt
import torch.nn.functional as F
import xlrd
import xlutils  #操作 Excel 文件的实用工具，如复制、分割、筛选等
import xlwt
#import vit_fusion

from torchinfo import summary



'''-------------一、SE模块-----------------------------'''


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        '''
        :param in_channels: 输入通道数
        :param out_channels:  输出通道数
        :param bilinear: 是否采用双线性插值，默认采用
        '''
        super(Up, self).__init__()
        if bilinear:
            # 双线性差值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = doubleConv(in_channels, out_channels, in_channels // 2)  # 拼接后为1024，经历第一个卷积后512
        else:
            # 转置卷积实现上采样
            # 输出通道数减半，宽高增加一倍
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = doubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 上采样
        x1 = self.up(x1)
        # 拼接
        x = torch.cat([x1, x2], dim=1)
        # 经历双卷积
        x = self.conv(x)
        return x


# 双卷积层
def doubleConv(in_channels, out_channels, mid_channels=None):
    '''
    :param in_channels: 输入通道数
    :param out_channels: 双卷积后输出的通道数
    :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
    :return:
    '''
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(mid_channels))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def down(in_channels, out_channels):
    # 池化 + 双卷积
    layer = []
    layer.append(nn.MaxPool2d(2, stride=2))
    layer.append(doubleConv(in_channels, out_channels))
    return nn.Sequential(*layer)


# 下采样
def outdown(in_channels=32,dow_rate=3,mid_channels=32):
    # 池化 + 双卷积
    layer = []
    for i in range(dow_rate):
        layer.append(nn.MaxPool2d(2, stride=2))
        layer.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False))
        layer.append(nn.BatchNorm2d(mid_channels))
        layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(1))
    layer.append(nn.LeakyReLU())#nn.LeakyReLU()

    return nn.Sequential(*layer)




# 整个网络架构
class U_net(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, base_channel=16,dow_rate=2):
        '''
        :param in_channels: 输入通道数，一般为3，即彩色图像
        :param out_channels: 输出通道数，即网络最后输出的通道数，一般为2，即进行2分类
        :param bilinear: 是否采用双线性插值来上采样，这里默认采取
        :param base_channel: 第一个卷积后的通道数，即64
        '''
        super(U_net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # 输入
        self.in_conv = doubleConv(self.in_channels, base_channel)
        # 下采样
        self.down1 = down(base_channel, base_channel * 2)  # 64,128
        self.down2 = down(base_channel * 2, base_channel * 4)  # 128,256
        self.down3 = down(base_channel * 4, base_channel * 8)  # 256,512
        # 最后一个下采样，通道数不翻倍（因为双线性差值，不会改变通道数的，为了可以简单拼接，就不改变通道数）
        # 当然，是否采取双线新差值，还是由我们自己决定
        factor = 2 if self.bilinear else 1
        self.down4 = down(base_channel * 8, base_channel * 16 // factor)  # 512,512
        # 上采样 + 拼接
        self.up1 = Up(base_channel * 16, base_channel * 8 // factor, self.bilinear)  # 1024(双卷积的输入),256（双卷积的输出）
        self.up2 = Up(base_channel * 8, base_channel * 4 // factor, self.bilinear)
        self.up3 = Up(base_channel * 4, base_channel * 2 // factor, self.bilinear)
        self.up4 = Up(base_channel * 2, base_channel, self.bilinear)
        # 输出
        #self.out = nn.Conv2d(in_channels=base_channel, out_channels=self.out_channels, kernel_size=1)
        self.out=outdown(base_channel*2,dow_rate=dow_rate)
    def forward(self, x):
        # x1 = self.in_conv(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # # 不要忘记拼接
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)             #100 16 128 128
        # x = self.out(x)
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 不要忘记拼接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
      #  x = self.up4(x, x1)  # 100 16 128 128
        x = self.out(x)
        x = nn.Flatten(start_dim=1)(x)

        return x



class mlp(nn.Module):
    def __init__(self,inarr,outarr):
        super(mlp,self).__init__()

        self.drop1 = nn.Dropout(p=0.5)
        self.Sigmoid = nn.Sigmoid()
        self.relu =nn.ReLU()
        self.bn = nn.LayerNorm(outarr)
      #  self.Batchnorm1=nn.BatchNorm2d()
        self.linear_test = nn.Linear(inarr,outarr)

    def mlp(self,x):

        x = nn.Flatten(start_dim=1)(x)
        x = self.linear_test(x)
        x = self.bn(x)
      #  x = self.Sigmoid(x)
        x = F.leaky_relu_(x)
        x  = self.drop1(x)
        return x

    def forward(self,x):
        x  = self.mlp(x)


        return x

class cnn_16(nn.Module):
    def __init__(self,in_channels, emb_dim):
        super(cnn_16,self).__init__()
        self.emb_dim=emb_dim
        self.conv1 = nn.Conv2d(in_channels, emb_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, stride=2, padding=1)
        self.norm=nn.LayerNorm(47)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.Sigmoid = nn.Sigmoid()
        self.relu =nn.ReLU()
        self.Batchnorm1=nn.BatchNorm2d(emb_dim)
        self.Batchnorm2 = nn.BatchNorm2d(emb_dim)
        self.linear_test = nn.Linear(256*emb_dim,256)
    def conv_encoder(self,x):
        x=self.conv1(x)
        x = self.Batchnorm1(x)
        x = self.relu(x)
        x=self.drop1(x)
        # x = self.conv2(x)
        # x = self.Batchnorm2(x)
        # x = self.relu(x)
        # x = self.drop2(x)
        return x
    def mlp(self,x):
        x = nn.Flatten(start_dim=1)(x)
        #x = nn.Linear(x.shape[-1], out_features).cuda()(x)
        x = self.linear_test(x)
        x = F.sigmoid(x)
        return x
    def forward(self,x):
        x = self.conv_encoder(x)#3通道
        x  = self.mlp(x)
        return x


class cnn(nn.Module):
    def __init__(self,in_channels, emb_dim):
        super(cnn,self).__init__()
        self.emb_dim=emb_dim
        self.conv1 = nn.Conv2d(in_channels, emb_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, stride=2, padding=1)
        self.norm=nn.LayerNorm(47)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.Sigmoid = nn.Sigmoid()
        self.relu =nn.ReLU()
        self.Batchnorm1=nn.BatchNorm2d(emb_dim)
        self.Batchnorm2 = nn.BatchNorm2d(emb_dim)
        self.linear_test = nn.Linear(1024*emb_dim,1024)
    def conv_encoder(self,x):
        x=self.conv1(x)
        x = self.Batchnorm1(x)
        x = self.relu(x)
        x=self.drop1(x)
        # x = self.conv2(x)
        # x = self.Batchnorm2(x)
        # x = self.relu(x)
        # x = self.drop2(x)
        return x
    def mlp(self,x):
        x = nn.Flatten(start_dim=1)(x)
        #x = nn.Linear(x.shape[-1], out_features).cuda()(x)
        x = self.linear_test(x)
        x = F.sigmoid(x)
        return x
    def forward(self,x):
        x = self.conv_encoder(x)#3通道
        x  = self.mlp(x)
        return x
    # @torch.no_grad
    # def feature_out(self,x):
    #     x=self.conv_encoder(x)
    #     return x

# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid

class cnn_big(nn.Module):
    def __init__(self,in_channels, emb_dim):
        super(cnn_big,self).__init__()
        self.emb_dim=emb_dim
        self.conv1 = nn.Conv2d(in_channels, emb_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, stride=2, padding=1)
        self.norm=nn.LayerNorm(47)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.Sigmoid = nn.Sigmoid()
        self.relu =nn.ReLU()
        self.Batchnorm1=nn.BatchNorm2d(emb_dim)
        self.Batchnorm2 = nn.BatchNorm2d(emb_dim)
        self.linear_test = nn.Linear(int(180000),1024)#115200 180000
    def conv_encoder(self,x):
        x=self.conv1(x)
        x = self.Batchnorm1(x)
        x = self.relu(x)
        x=self.drop1(x)
        x = self.conv2(x)
        x = self.Batchnorm2(x)
        x = self.relu(x)
        x = self.drop2(x)
        return x
    def mlp(self,x):
        x = nn.Flatten(start_dim=1)(x)
        #x = nn.Linear(x.shape[-1], out_features).cuda()(x)
        x = self.linear_test(x)
        x = F.sigmoid(x)
        return x
    def forward(self,x):
        x = self.conv_encoder(x)#3通道
        x  = self.mlp(x)
        return x
    # @torch.no_grad
    # def feature_out(self,x):
    #     x=self.conv_encoder(x)
    #     return x

# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid

class cnn_big_16(nn.Module):
    def __init__(self,in_channels, emb_dim):
        super(cnn_big_16,self).__init__()
        self.emb_dim=emb_dim
        self.conv1 = nn.Conv2d(in_channels, emb_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, stride=2, padding=1)
        self.norm=nn.LayerNorm(47)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.Sigmoid = nn.Sigmoid()
        self.relu =nn.ReLU()
        self.Batchnorm1=nn.BatchNorm2d(emb_dim)
        self.Batchnorm2 = nn.BatchNorm2d(emb_dim)
        self.linear_test = nn.Linear(int(80000),256)#115200 180000
    def conv_encoder(self,x):
        x=self.conv1(x)
        x = self.Batchnorm1(x)
        x = self.relu(x)
        x=self.drop1(x)
        x = self.conv2(x)
        x = self.Batchnorm2(x)
        x = self.relu(x)
        x = self.drop2(x)
        return x
    def mlp(self,x):
        x = nn.Flatten(start_dim=1)(x)
        #x = nn.Linear(x.shape[-1], out_features).cuda()(x)
        x = self.linear_test(x)
        x = F.sigmoid(x)
        return x
    def forward(self,x):
        x = self.conv_encoder(x)#3通道
        x  = self.mlp(x)
        return x
class mlp_net(nn.Module):
    def __init__(self, emb_dim):
        super(mlp_net,self).__init__()
        self.emb_dim=emb_dim
        self.linear_1 = nn.Linear(emb_dim*emb_dim,emb_dim * emb_dim)
        self.linear_2 = nn.Linear(emb_dim * emb_dim, 4096)

    def mlp(self,x):

        x = nn.Flatten(start_dim=1)(x)
        x = self.linear_1(x)
        x = F.sigmoid(x)
        x = self.linear_2(x)
        x = F.sigmoid(x)
        return x

    def forward(self,x):

        x  = self.mlp(x)
        return x