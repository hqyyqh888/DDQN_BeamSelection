# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 19:26:44 2020

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ConvBn(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        )
        
    def forward(self, x):
        return self.net(x)


class ConvBnPrelu(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn(in_c, out_c, kernel, stride, padding, groups),
            nn.PReLU(out_c)
        )

    def forward(self, x):
        return self.net(x)


class DepthWise(nn.Module):

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnPrelu(in_c, groups, kernel=(1, 1), stride=1, padding=0),
            ConvBnPrelu(groups, groups, kernel=kernel, stride=stride, padding=padding, groups=groups),
            ConvBn(groups, out_c, kernel=(1, 1), stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)


class DepthWiseRes(nn.Module):
    """DepthWise with Residual"""

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = DepthWise(in_c, out_c, kernel, stride, padding, groups)

    def forward(self, x):
        return self.net(x) + x


class MultiDepthWiseRes(nn.Module):

    def __init__(self, num_block, channels, kernel=(3, 3), stride=1, padding=1, groups=1):
        super().__init__()

        self.net = nn.Sequential(*[
            DepthWiseRes(channels, channels, kernel, stride, padding, groups)
            for _ in range(num_block)
        ])

    def forward(self, x):
        return self.net(x)
    
# class Net_dueling_DDQN(nn.Module):
#     def __init__(self,input_dim, output_dim):
#         """channel size M = search space number N = RF chain number"""  
#         self.channel = input_dim[0]
#         self.M = input_dim[1]
#         self.K = input_dim[2]
  
#         super().__init__()
#         self.res1 = nn.Linear(self.M*self.K*self.channel, self.K, bias=False)
#         self.conv1 = ConvBnPrelu(3, 12, kernel=(3, 3), stride=1, padding=1)
#         self.conv2 = ConvBn(12, 12, kernel=(3, 3), stride=1, padding=1, groups=12)
#         self.conv3 = DepthWise(12, 12, kernel=(3, 3), stride=2, padding=1, groups=12)
#         self.conv4 = MultiDepthWiseRes(num_block=6, channels=12, kernel=3, stride=1, padding=1, groups=12)
#         self.conv5 = ConvBn(12, 12, groups=12, kernel=(3, 3))
#         self.flatten = Flatten()
#         self.linear = nn.Linear(312, output_dim, bias=False)
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)
#         out = self.flatten(out)
#         out = self.linear(out)
#         return out

class Net_dueling_DDQN(nn.Module):
    def __init__(self,input_dim, output_dim):
        """channel size M = search space number N = RF chain number"""  
        self.channel = input_dim[0]
        self.M = input_dim[1]
        self.K = input_dim[2]
  
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (2,2), stride=1)
        self.BN1 =  nn.BatchNorm2d(6)
        self.conv2 = ConvBn(6, 8,(2, 2), stride=1)
        self.BN2 = nn.BatchNorm2d((8))
        self.flatten = Flatten()
        self.linear1 = nn.Linear(1344, 800)
        self.linear2 = nn.Linear(800, 300)
        self.linear3 = nn.Linear(300, output_dim, bias=False)       
        
        
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.BN1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.BN2(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out
    



class Net_actor(nn.Module):
    def __init__(self, M_bar, K):
        """channel size M = search space number N = RF chain number"""  
        self.M = M_bar
        self.K = K
        output_dim = M_bar
        super().__init__()
        self.res1 = nn.Linear(self.M*self.K*3, self.K, bias=False)
        self.conv1 = ConvBnPrelu(3, 12, kernel=(3, 3), stride=1, padding=1)
        self.conv2 = ConvBn(12, 12, kernel=(3, 3), stride=1, padding=1, groups=12)
        self.conv3 = DepthWise(12, 12, kernel=(3, 3), stride=2, padding=1, groups=12)
        self.conv4 = MultiDepthWiseRes(num_block=4, channels=12, kernel=3, stride=1, padding=1, groups=12)
        self.conv5 = ConvBn(12, 12, groups=12, kernel=(3, 3))
        self.flatten = Flatten()
        self.linear = nn.Linear(1296, output_dim, bias=False)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out
    
class Net_critic(nn.Module):
    def __init__(self, M_bar, K, output_dim=1):
        """channel size M = search space number N = RF chain number"""  
        self.M = M_bar
        self.K = K
        super().__init__()
        self.res1 = nn.Linear(self.M*self.K*3, self.K, bias=False)
        
        self.conv1 = ConvBnPrelu(3, 12, kernel=(3, 3), stride=1, padding=1)
        self.conv2 = ConvBn(12, 12, kernel=(3, 3), stride=1, padding=1, groups=12)
        self.conv3 = DepthWise(12, 12, kernel=(3, 3), stride=2, padding=1, groups=12)
        self.conv4 = MultiDepthWiseRes(num_block=4, channels=12, kernel=3, stride=1, padding=1, groups=12)
        self.conv5 = ConvBn(12, 12, groups=12, kernel=(3, 3))
        self.flatten = Flatten()
        self.linear = nn.Linear(1008, output_dim, bias=False)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out
    
