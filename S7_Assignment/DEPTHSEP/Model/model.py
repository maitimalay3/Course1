# -*- coding: utf-8 -*-
"""Mtest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14oUlZI4c5D6iLUCb3qcQosha868Vvemy
"""

from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from ModelStats import ModelStats
from tqdm import tqdm_notebook, tnrange
import torch.nn.functional as F
import torch

#sys.path.append('/content/drive//MyDrive/CVMODEL/DEPTHSEP/Model/')
#import ModelTrainer

class Train:
  def __init__(self, model, dataloader, optimizer, stats, scheduler=None, L1lambda = 0, LossFunction='CrossEntropyLoss'):
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.stats = stats
    self.L1lambda = L1lambda
    self.Loss=LossFunction

  def run(self):
    self.model.train()
    pbar = tqdm_notebook(self.dataloader)
    for data, target in pbar:
      # get samples
      data, target = data.to(self.model.device), target.to(self.model.device)

      # Init
      self.optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = self.model(data)

      # Calculate loss
      loss = self.Loss(y_pred, target)

      #Implementing L1 regularization
      if self.L1lambda > 0:
        loss += L1_Loss(Model=self.Model, L1lambda=self.L1lambda)

      # Backpropagation
      loss.backward()
      self.optimizer.step()

      # Update pbar-tqdm
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct = pred.eq(target.view_as(pred)).sum().item()
      lr = self.scheduler.get_last_lr()[0] if self.scheduler else (self.optimizer.param_groups[0]['lr'])
      self.stats.add_batch_train_stats(loss.item(), correct, len(data), lr)
      pbar.set_description(self.stats.get_latest_batch_desc())
      if self.scheduler:
        self.scheduler.step()
        
    def L1_Loss(Model, L1lambda):
        reg_loss=0
        l1_crit = torch.nn.L1Loss(size_average=False)
        for param in Model.parameters():
            target = torch.zeros_like(param)
            reg_loss += l1_crit(param, target)    
        return L1lambda*reg_loss
        
        
class Test:
  def __init__(self, model, dataloader, stats, LossFunction):
    self.model = model
    self.dataloader = dataloader
    self.stats = stats
    self.Loss=LossFunction

  def run(self):
    self.model.eval()
    loss=0
    with torch.no_grad():
        for data, target in self.dataloader:
            data, target = data.to(self.model.device), target.to(self.model.device)
            output = self.model(data)
            loss += self.Loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            self.stats.add_batch_test_stats(loss, correct, len(data))

class ModelTrainer:
  def __init__(self, model, optimizer, train_loader, test_loader, statspath, scheduler=None, batch_scheduler=False, L1lambda = 0, LossType='CrossEntropyLoss'):
    self.model = model
    self.scheduler = scheduler
    self.batch_scheduler = batch_scheduler
    self.optimizer = optimizer
    self.stats = ModelStats(model, statspath)
    self.LossFunction = self.LossFunction(LossType)
    self.train = Train(model, train_loader, optimizer, self.stats, self.scheduler if self.scheduler and self.batch_scheduler else None, L1lambda, self.LossFunction)
    self.test = Test(model, test_loader, self.stats, self.LossFunction)
    
  def run(self, epochs=10):
    pbar = tqdm_notebook(range(1, epochs+1), desc="Epochs")
    for epoch in pbar:
      self.train.run()
      self.test.run()
      self.stats.next_epoch(self.scheduler.get_last_lr()[0] if self.scheduler else 0)
      pbar.write(self.stats.get_epoch_desc())
      if self.scheduler and not self.batch_scheduler:
        self.scheduler.step()
      if self.scheduler:
        pbar.write(f"Learning Rate = {self.scheduler.get_last_lr()[0]:0.6f}")
    # save stats for later lookup
    self.stats.save()
    
  def LossFunction(self,Loss_Type="CrossEntropyLoss"):
    """L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss."""
    if Loss_Type=="CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()

class Net(nn.Module, ModelTrainer):
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

    def gotrain(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0, LossType='CrossEntropyLoss'):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, L1lambda, LossType)
      self.trainer.run(epochs)

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