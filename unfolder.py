# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:00:20 2021

@author: Administrator
"""


import torch
from complex_matrix import *
import numpy as np
from model import * 
import torch.optim as optim
from config import Config
from WMMSE import WMMSE

class Unfolder():
    def __init__(self,config):
        self.K = config.K
        self.d = 1
        self.Nr = 1
        self.Nt = config.beam_num
        self.Pmax = np.power(10,config.snr/10)
        self.sigma = 1
        self.layer_num = 5
        
        self.device = config.device
        self.dtype = torch.float32
        self.unfolding_network = model(config,self.layer_num).to(device=self.device,dtype=self.dtype) 
        self.optimizer = optim.SGD(self.unfolding_network.parameters(),lr=0.001)
         
      
    def loss_f(self,H,V,K):
        _,K,Nr,Nt=H.size()
        I1 = torch.eye(Nr)
        I2 = torch.zeros((Nr,Nr))
        I = mcat(I1,I2)
        temp = []
        temp1 = []
        temp2 = []
        loss = 0
        for i in range(K):
            temp1.append([])
            temp2.append([])
            temp1[i].append(0)
            temp2[i].append(0)
            for j in range(K):
                if j != i:
                    temp1[i].append(temp1[i][j]+cmul(cmul(cmul(H[:,i,:,:],V[:,j,:,:]),conjT(V[:,j,:,:])),conjT(H[:,i,:,:])))
                else:
                    temp1[i].append(temp1[i][j])
                temp2[i].append(temp2[i][j]+self.sigma/self.Pmax*cmul1(mcat(torch.trace(cmul(V[:,j,:,:],conjT(V[:,j,:,:]))[0,:,:]),torch.trace(cmul(V[:,j,:,:],conjT(V[:,j,:,:]))[1,:,:])),I))
            temp.append(cinv((temp1[i][K]+temp2[i][K])))        
            loss -= torch.log2(complex_det(I+cmul(cmul(cmul(cmul(H[:,i,:,:],V[:,i,:,:]),conjT(V[:,i,:,:])),conjT(H[:,i,:,:])),temp[i]))[0])           
        return loss
    
    def train_network(self, H_set, epochs, batch_num=5):
        for e in range(epochs):
            H_train = H_set
            V_train = self.InitialV(len(H_train))
            i = 0
            loss = 0
            for H, V in zip(H_train, V_train):
                H = H.to(dtype=self.dtype,device=self.device)
                self.unfolding_network.train()  # put model to training mode
                V_final = self.unfolding_network(V,H)
                loss += self.loss_f(H,V_final,self.K)/batch_num
                i +=1
                if i%batch_num==0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    for name, parms in self.unfolding_network.named_parameters():
                        if parms.grad is not None:                    
                            parms.grad = parms.grad / (torch.norm(parms.grad)+1e-32)
                    self.optimizer.step()
                    loss = 0
                    
    def InitialV(self,length):
        PT = self.Pmax
        sigma = self.sigma
        V_set=[]
        for i in range(length):
            V = np.random.randn(self.K,self.Nt,self.Nr)+1j*np.random.randn(self.K,self.Nt,self.Nr)
            V = c2m(V).to(dtype=self.dtype,device=self.device)
            V_set.append(V)
        return V_set
    
    def InitialH(self,length):
        H_set=[]
        for i in range(length):
            H = np.random.randn(self.K,self.Nr,self.Nt)+1j*np.random.randn(self.K,self.Nr,self.Nt)
            H = c2m(H).to(dtype=self.dtype,device=self.device)
            H_set.append(H)
        return H_set
    
    def self_train_network(self,epoch,sample_num = 100, batch_num =5):
        
        print('start self training unfolding network')
        for e in range(epoch):
            H_set = self.InitialH(sample_num)
            print('epoch %d ======',e)
            self.train_network(H_set,1,batch_num)
            self.test_network(test_sample_num=20)
        print('self training is over')
    
    def forward(self, H):
        V = np.random.randn(self.K,self.Nt,self.Nr)+1j*np.random.randn(self.K,self.Nt,self.Nr)
        V = c2m(V).to(dtype=self.dtype,device=self.device)
        self.unfolding_network.eval()
        V_final = self.unfolding_network(V,H)
        rate = self.loss_f(H,V_final,self.K)  
        
        return rate.detach()
        
    def test_network(self,test_sample_num):
        H_test = self.InitialH(test_sample_num)
        V_test = self.InitialV(len(H_test))
        loss = 0
        true_value = 0
        for H, V in zip(H_test, V_test):
            V_final = self.unfolding_network(V,H)
            loss += self.loss_f(H,V_final,self.K)/test_sample_num  
            H = torch.squeeze(H)
            true_value += WMMSE(conjT(H),self.Pmax,self.sigma)/test_sample_num   

    
