# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:24:17 2021

@author: mamh4s
"""

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    """
    Base network that defines helper functions, summary and mapping to device
    """
    def conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode)]

    def activate(self, l, out_channels, bn=True, dropout=0, relu=True):
      if relu:
        l.append(nn.ReLU())
      if bn:
        l.append(nn.BatchNorm2d(out_channels))
      if dropout>0:
        l.append(nn.Dropout(dropout))  
      return nn.Sequential(*l)

    def create_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode), out_channels, bn, dropout, relu)

    def __init__(self, name="Model"):
        super(Net, self).__init__()
        self.trainer = None
        self.name = name

    def summary(self, input_size): #input_size=(1, 28, 28)
      summary(self, input_size=input_size)

    #def gotrain(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0, LossType='CrossEntropyLoss'):
    #  self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, L1lambda, LossType)
     # self.trainer.run(epochs)

    def stats(self):
      return self.trainer.stats if self.trainer else None

class Cfar10Net(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(Cfar10Net, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 32, dropout=dropout_value)
        self.conv2 = self.create_conv2d(32, 32, dropout=dropout_value)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = self.create_conv2d(32, 64, padding=2, dilation=2, dropout=dropout_value)
        self.conv4 = self.create_conv2d(64,64, groups=64, dropout=dropout_value)
        self.conv5 = self.create_conv2d(64,128, dropout=dropout_value)
        self.conv6 = self.create_conv2d(128,128, dropout=dropout_value)
        self.conv7 = self.create_conv2d(128, 256, dropout=dropout_value)
        self.dconv1 = self.create_conv2d(16, 32, dilation=2, padding=2)
        self.conv8 = self.create_conv2d(256, 10, kernel_size=(1,1),padding=0, bn=False, relu=False)
        self.gap = nn.AvgPool2d(kernel_size=(3,3))
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool1(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)