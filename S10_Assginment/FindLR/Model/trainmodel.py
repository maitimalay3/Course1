# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:06:06 2021

@author: mamh4s
"""

from tqdm import tqdm_notebook, tqdm, tnrange
from ModelStats import ModelStats
import torch.nn.functional as F
import torch


class Train:
  def __init__(self, model, dataloader, optimizer, stats, scheduler=None, batch_scheduler=False, L1lambda = 0, LossFunction='CrossEntropyLoss'):
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.stats = stats
    self.L1lambda = L1lambda
    self.Loss=LossFunction
    self.batch_scheduler = batch_scheduler

  def run(self):
    self.model.train()
    pbar = tqdm(self.dataloader)
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
        loss += self.L1_Loss(Model=self.model, L1lambda=self.L1lambda)

      # Backpropagation
      loss.backward()
      self.optimizer.step()

      # Update pbar-tqdm
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct = pred.eq(target.view_as(pred)).sum().item()
      lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr'] 
      self.stats.add_batch_train_stats(loss.item(), correct, len(data), lr)
      pbar.set_description(self.stats.get_latest_batch_desc())
      if self.scheduler:
        flag = self.batch_scheduler.best_lr(self.stats,loss, lr)
        if flag:tqdm._instances.clear();break;
        self.scheduler.step()
        
  def L1_Loss(self, Model, L1lambda):
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
    self.train = Train(model, train_loader, optimizer, self.stats, self.scheduler if self.scheduler and self.batch_scheduler else None, self.batch_scheduler, L1lambda, self.LossFunction)
    self.test = Test(model, test_loader, self.stats, self.LossFunction)
    
  def run(self, epochs=10):
    pbar = tqdm_notebook(range(1, epochs+1), desc="Epochs")
    for epoch in pbar:
      self.train.run()
      self.test.run()
      try:
        scheduler_lr = self.scheduler.get_last_lr()[0]
      except:
        scheduler_lr = self.optimizer.param_groups[0]['lr']
      self.stats.next_epoch(scheduler_lr if self.scheduler else 0)
      pbar.write(self.stats.get_epoch_desc())
      if self.scheduler and not self.batch_scheduler:
        try:
            self.scheduler.step()
        except:
            self.scheduler.step(self.stats.avg_test_loss[-1])
      if self.scheduler:
        pbar.write(f"Learning Rate = {scheduler_lr:0.6f}")
    # save stats for later lookup
    self.stats.save()
    
  def LossFunction(self,Loss_Type="CrossEntropyLoss"):
    """L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss."""
    if Loss_Type=="CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()