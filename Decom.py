import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F


class decompose_net(nn.Module):

    def __init__(self):
        super(decompose_net,self).__init__()
        number_f=64
        self.relu=nn.PReLU()
        self.conv1=nn.Conv2d(3,number_f,3,1,1,bias=True)
        self.conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(number_f, 6, 3, 1, 1, bias=True)
        self.norm=nn.BatchNorm2d(4,affine=True)
    def forward(self,x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv2(x2))
        x4 = self.relu(self.conv2(x3))
        x5 = self.relu(self.conv2(x4))
        x_out =F.sigmoid((self.conv3(x5)))
        x_high_frequency = torch.cat([torch.unsqueeze(x_out[:, 0, :, :], 1), torch.unsqueeze(x_out[:, 1, :, :], 1),torch.unsqueeze(x_out[:, 2, :, :], 1)], 1)
        x_low_frequency = torch.cat([torch.unsqueeze(x_out[:, 3, :, :], 1), torch.unsqueeze(x_out[:, 4, :, :], 1),torch.unsqueeze(x_out[:, 5, :, :], 1)], 1)
        return x_high_frequency,x_low_frequency
