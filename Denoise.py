import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class Denois_net(nn.Module):
    def __init__(self):
        super(Denois_net, self).__init__()
        self.relu = nn.PReLU()
        number_f = 64
        self.conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv1_1 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv1_2 = nn.Conv2d(number_f, 3, 1, 1, 0, bias=True)
        self.norm64 = nn.BatchNorm2d(number_f, affine=True)
        self.norm = nn.BatchNorm2d(3, affine=True)
    def forward(self,x):
        x1 = self.norm64(self.relu(self.conv1(x)))
        # residual
        x1_1 = self.relu(self.conv1_1(x1))
        x1_2 = self.relu(self.conv1_1(x1_1))
        x1_3 = self.relu(self.conv1_2(x1 + x1_2))

        x2 = self.norm64(self.relu(self.conv2(x1)))
        x2_1 =  self.relu(self.conv1_1(x2))
        x2_2 =  self.relu(self.conv1_1(x2_1))
        x2_3 =  self.relu(self.conv1_2(x2_2 + x2))

        x3 = self.norm64(self.relu(self.conv2(x2)))
        x3_1 =  self.relu(self.conv1_1(x3))
        x3_2 =  self.relu(self.conv1_1(x3_1))
        x3_3 =  self.relu(self.conv1_2(x3_2 + x3))

        x4 = self.norm64(self.relu(self.conv2(x3)))
        x4_1 = self.relu(self.conv1_1(x4))
        x4_2 = self.relu(self.conv1_1(x4_1))
        x4_3 = self.relu(self.conv1_2(x4_2 + x4))

        x5 = self.norm64(self.relu(self.conv2(x4)))
        x5_1 = self.relu(self.conv1_1(x5))
        x5_2 = self.relu(self.conv1_1(x5_1))
        x5_3 = self.relu(self.conv1_2(x5_2 + x5))

        x_out = x1_3 + x3_3 + x2_3+ x4_3 + x5_3

        return x_out