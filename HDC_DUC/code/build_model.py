
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision
from torchvision import models
from torchvision import datasets, transforms


#import torchvision.datasets as dset
#import torchvision.transforms as T
import torch.nn.functional as F

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import copy


class DUC(nn.Module):
  #d=downsample_factor, L=num_of_classes
  def __init__(self, in_channels, d, L ):
    super(DUC, self).__init__()
    out_channels = (d**2)*L
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3))
    self.BN = nn.BatchNorm2d(out_channels, affine = False) #Should affine be True only?
    self.pixel_shuffle = nn.PixelShuffle(d)
    
  def forward(self,x):
    x = self.conv(x)
    x = self.BN(x)
    x = F.relu(x)
    x = self.pixel_shuffle(x)
    return x

class ResNetwithHDCDUC(nn.Module):
  def __init__(self, L, pretrained=True):
    super(ResNetwithHDCDUC, self).__init__()
    model = torchvision.models.resnet101(pretrained=True)
    self.res1 = nn.Sequential(*list(model.children())[0:3])
    self.res2 = nn.Sequential(*list(model.children())[4])
    self.res3 = nn.Sequential(*list(model.children())[5])
    self.res4 = nn.Sequential(*list(model.children())[6])
    self.res5 = nn.Sequential(*list(model.children())[7])
    self.avg_pool = list(model.children())[8]
    #self.avg_pool = nn.Sequential(*list(model.children())[8])
    self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=1)
    
    layer4_group_config = [1, 2, 5, 9]
    for i in range(len(self.res4)):
            self.res4[i].conv2.dilation = (layer4_group_config[i % 4], layer4_group_config[i % 4])
            self.res4[i].conv2.padding = (layer4_group_config[i % 4], layer4_group_config[i % 4])
    layer5_group_config = [5, 9, 17]
    for i in range(len(self.res5)):
            self.res5[i].conv2.dilation = (layer5_group_config[i], layer5_group_config[i])
            self.res5[i].conv2.padding = (layer5_group_config[i], layer5_group_config[i])
    
    
    in_channels = 2048
    d = 32
    self.duc_func = DUC(in_channels, d, L=L)
    
  def forward(self, x):
    x1 = self.res1(x)
    x2 = self.res2(x1)
    x3 = self.res3(x2)
    x4 = self.res4(x3)
    x5 = self.res5(x4)
    x6 = self.avg_pool(x5)
    #x8 = self.max_pool(x6)
    #in_channels = 2048
    #d = float(x.shape[2]/x5.shape[2])
    #duc_func = DUC(in_channels, d, 1)
    x7 = self.duc_func(x6)

    return x7
