import torch 
import torch.nn as nn
import numpy as np


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 50, kernel_size=(3,3), padding=1), 
        nn.ReLU(),
        )
    self.conv2 = nn.Sequential(
        nn.Conv2d(50, 100, kernel_size=(3,3), padding=1), 
        nn.ReLU(),
        )
    self.conv3 = nn.Sequential(
        nn.Conv2d(100, 100, kernel_size=(3,3), padding=1), 
        nn.ReLU(),
        )
    self.conv4 = nn.Sequential(
        nn.Conv2d(100, 100, kernel_size=(3,3), padding=1), 
        nn.ReLU(),
        )
    self.linear_relu = nn.Sequential(
        nn.Linear(1600,100),
        nn.ReLU(),
    )
    self.linear = nn.Sequential(
        nn.Linear(100, 10)
    )
    self.maxpool1 = nn.MaxPool2d((2,2))
    self.maxpool2 = nn.MaxPool2d((2,2))
    self.maxpool3 = nn.MaxPool2d((2,2))
    
    #Replace with your code
    
  def forward(self, x):
    out1 = self.maxpool1(self.conv1(x))
    out2 = self.maxpool2(self.conv2(out1))
    out3 = self.maxpool3(self.conv3(out2))
    out4 = self.conv4(out3) # error
    out_reshape = out4.reshape(out4.shape[0], out4.shape[1]*out4.shape[2]*out4.shape[3])
    out5 = self.linear_relu(out_reshape)
    out6 = self.linear(out5)
    return out6 #Replace with your code
