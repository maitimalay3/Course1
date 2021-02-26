from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelTrainer import ModelTrainer
import Resnet as rn

class Net(nn.Module):
    """
    Base network that defines helper functions, summary and mapping to device
    """
    def conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode)]
      
    def separable_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), bias=bias)]

    def activate(self, l, out_channels, bn=True, dropout=0, relu=True):
      if bn:
        l.append(nn.BatchNorm2d(out_channels))
      if relu:
        l.append(nn.ReLU())
      if dropout>0:
        l.append(nn.Dropout(dropout))  
      return nn.Sequential(*l)

    def create_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode), out_channels, bn, dropout, relu)

    def create_depthwise_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.separable_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
                 out_channels, bn, dropout, relu)
                 
    def __init__(self, name="Model"):
        super(Net, self).__init__()
        self.trainer = None
        self.name = name

    def summary(self, input_size): #input_size=(1, 28, 28)
      summary(self, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, dataloader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0, LossType='CrossEntropyLoss'):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, dataloader, statspath, scheduler, batch_scheduler, L1lambda, LossType)
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
        
class Resnet_Architecture(Net):
    def __init__(self, name="Model", Model_Layers=18, num_classes=10):
        super(Resnet_Architecture, self).__init__(name)
        self.Model_Layers=Model_Layers
        self.num_classes=num_classes
        
        if self.Model_Layers==18:
            self.resnet_arch = rn.ResNet(rn.BasicBlock, [2,2,2,2],self.num_classes)
        elif self.Model_Layers==34:
            self.resnet_arch = rn.ResNet(rn.BasicBlock, [3,4,6,3],self.num_classes)
        elif self.Model_Layers==50:
            self.resnet_arch = rn.ResNet(rn.Bottleneck, [3,4,6,3],self.num_classes)
        elif self.Model_Layers==101:
            self.resnet_arch = rn.ResNet(rn.Bottleneck, [3,4,23,3],self.num_classes)
        elif self.Model_Layers==152:
            self.resnet_arch = rn.ResNet(rn.Bottleneck, [3,8,36,3],self.num_classes)
        else:
            raise ValueError('Please choose the value from 18,34,50,101,152.')
    
    def forward(self,x):
        return self.resnet_arch(x)
        
    def test(self):
        net = rn.ResNet18()
        y = net(torch.randn(1,3,32,32))
        print(y.size())

    # test()
    
class Resnet_Custom_Architecture(Net):
    def __init__(self, name="Model", num_classes=10):
        super(Resnet_Custom_Architecture, self).__init__(name)
        
        self.layer0=self.create_conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.layer1=self.ResBlock(in_channels=64, out_channels=128, kernel_size=(3,3),padding=1, bn=True, dropout=0, relu=True)
        self.Resb1=self.ResBlock(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, bn=True, dropout=0, relu=True, rep=2)
        self.layer2=self.ResBlock(in_channels=128, out_channels=256, kernel_size=(3,3),padding=1,bn=True, dropout=0, relu=True)
        self.layer3=self.ResBlock(in_channels=256, out_channels=512, kernel_size=(3,3),padding=1,bn=True, dropout=0, relu=True)
        self.Resb2=self.ResBlock(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bn=True, dropout=0, relu=True, rep=2)
        self.pool=nn.MaxPool2d(4)
        self.linear = nn.Linear(512, num_classes, bias=False)
        
    def ResBlock(self, in_channels, out_channels, kernel_size, padding=1, bn=False, dropout=0, relu=False, rep=0):
        layer=[]
        if rep==0:
            layer.append(self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)+[nn.MaxPool2d(2)],out_channels, bn=bn, dropout=dropout, relu=relu))
        for i in range(rep):
            layer.append(self.create_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,bn=bn, dropout=dropout, relu=relu))
        return nn.Sequential(*layer)
        
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out= self.Resb1(out)+out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.Resb2(out)+out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(out.size(0), -1)
        return F.log_softmax(out, dim=-1)