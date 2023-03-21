import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layer1 = nn.Sequential(
        nn.Conv2d(3, 50, kernel_size=(3,3), padding=1), 
        nn.ReLU(),
        )
    self.conv_layer2 = nn.Sequential(
        nn.Conv2d(50, 100, kernel_size=(3,3), padding=1), 
        nn.ReLU(),
        )
    # conv_layer3 replaced with residual layer
    self.residual_layer = nn.Sequential(
        nn.Conv2d(100, 100, kernel_size=(3,3), padding=1), 
        nn.ReLU(),
        nn.Conv2d(100, 100, kernel_size=(3,3), padding=1), 
        )
    self.linear_relu = nn.Sequential(
        nn.Linear(1600,100),
        nn.ReLU(),
    )
    self.linear = nn.Sequential(
        nn.Linear(100, 10)
    )
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d((2,2))

  def forward(self, x):
    out1 = self.maxpool(self.conv_layer1(x))
    out2 = self.maxpool(self.conv_layer2(out1))
    out3 = self.maxpool(self.relu(self.residual_layer(out2)+out2)) # residual block 1
    out4 = self.relu(self.residual_layer(out3)+out3) # residual block 2
    out_reshape = out4.reshape(out4.shape[0], out4.shape[1]*out4.shape[2]*out4.shape[3])  # Reshape for linear layers
    out5 = self.linear_relu(out_reshape)
    out6 = self.linear(out5)
    return out6 
