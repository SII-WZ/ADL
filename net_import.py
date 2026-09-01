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
import xlutils  
import xlwt
#import vit_fusion

from torchinfo import summary




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
        x = self.conv_encoder(x)
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
        x = self.conv_encoder(x)
        x  = self.mlp(x)
        return x
    # @torch.no_grad
    # def feature_out(self,x):
    #     x=self.conv_encoder(x)
    #     return x


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
        x = self.conv_encoder(x)
        x  = self.mlp(x)
        return x
    # @torch.no_grad
    # def feature_out(self,x):
    #     x=self.conv_encoder(x)
    #     return x


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
        self.linear_test = nn.Linear(int(80000),256)
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
        x = self.conv_encoder(x)
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
