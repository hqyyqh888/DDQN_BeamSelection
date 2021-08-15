import torch
from complex_matrix import *
import numpy as np
import torch.nn as nn
import math
#总模型
class model(nn.Module):
    def __init__(self,config,layer_num):
        super(model,self).__init__()
        self.K = config.K
        self.d = 1
        self.Nr = 1
        self.Nt = config.beam_num
        self.Pmax = np.power(10,config.snr/10)
        self.sigma = 1
        self.n = layer_num
        
        self.device = config.device
        self.MidLayer_f = {}
        
        for i in range(self.n):
            self.MidLayer_f["%d"%(i)] = MidLayer(self.K,self.Nr,self.Nt,self.d,self.Pmax,self.sigma)
        self.MidLayer_f = nn.ModuleDict(self.MidLayer_f)
        self.FinalLayer_f = FinalLayer(self.K,self.Nr,self.Nt,self.d,self.Pmax,self.sigma)
    def forward(self,V,H):
        V_c = {}
        V_c['0'] = V
        for i in range(self.n):
            V_c['%d'%(i+1)] = self.MidLayer_f["%d"%(i)](V_c['%d'%i],H)  #将n个中间层串联
        return self.FinalLayer_f(V_c['%d'%self.n],H)
#中间层

class MidLayer(nn.Module):
    def __init__(self,K,Nr,Nt,d,Pmax,sigma):
        super(MidLayer,self).__init__()
        self.K = K
        self.Nr = Nr
        self.Nt = Nt
        self.d = d
        self.Pmax = Pmax
        self.sigma = sigma
        
        self.U_f = U(self.K,self.Nr,self.d)
        self.W_f = W(self.K,self.d)
        self.V_f = V(self.K,self.Nt,self.d)
    def forward(self,V,H):
        A_new = A_update(self.K,self.Nr,V,H,self.Pmax,self.sigma)
        U_new = self.U_f(H,V,A_new)
        E_new = E_update(self.K,self.d,U_new,V,H)
        W_new = self.W_f(E_new)
        B_new = B_update(self.K,self.Nt,U_new,W_new,H,self.Pmax,self.sigma)
        V_new = self.V_f(H,U_new,B_new,W_new)
        V_new = Normalize_f(self.K,V_new,self.Pmax)
        return V_new

 #最后一层 
class FinalLayer(nn.Module):
    def __init__(self,K,Nr,Nt,d,Pmax,sigma):
        super(FinalLayer,self).__init__()
        self.K = K
        self.Nt = Nt
        self.Nr = Nr 
        self.d = d 
        self.Pmax = Pmax
        self.sigma = sigma
        
        self.U_f = U(self.K,self.Nr,self.d)
        self.W_f = W(self.K,self.d)
    def forward(self,V,H):
        A_new = A_update(self.K,self.Nr,V,H,self.Pmax,self.sigma)
        U_new = self.U_f(H,V,A_new)
        E_new = E_update(self.K,self.d,U_new,V,H)
        W_new = self.W_f(E_new)
        B_new = B_update(self.K,self.Nt,U_new,W_new,H,self.Pmax,self.sigma)
        V_new = FinalV(self.K,self.Nt,self.d,H,U_new,B_new,W_new)
        V_new = Normalize_f(self.K,V_new,self.Pmax)
        return V_new



def A_update(K,Nr,V,H,Pmax,sigma):
    A = torch.zeros((2,K,Nr,Nr))
    I1 = torch.eye(Nr)
    I2 = torch.zeros((Nr,Nr))
    I = mcat(I1,I2)
    temp1 = torch.zeros(2)
    temp2 = torch.zeros_like(I)
    for i in range(K):
        temp1 = 0
        temp2 = torch.zeros_like(I)
        for j in range(K):
            temp1 = temp1 + mcat(torch.trace(cmul(V[:,j,:,:],conjT(V[:,j,:,:]))[0,:,:]),torch.trace(cmul(V[:,j,:,:],conjT(V[:,j,:,:]))[1,:,:]))
            temp2 = temp2 + cmul(cmul(cmul(H[:,i,:,:],V[:,j,:,:]),conjT(V[:,j,:,:])),conjT(H[:,i,:,:]))
        A[:,i,:,:] = sigma/Pmax*cmul1(temp1,I) + temp2
    
    return A
    
def B_update(K,Nt,U_new,W_new,H,Pmax,sigma):
    B = torch.zeros((2,K,Nt,Nt))
    temp1 = 0
    temp2 = 0
    I1 = torch.eye(Nt)
    I2 = torch.zeros((Nt,Nt))
    I = mcat(I1,I2)
    for i in range(K):
        temp1 += sigma/Pmax*cmul1(mcat(torch.trace(cmul(cmul(U_new[:,i,:,:],W_new[:,i,:,:]),conjT(U_new[:,i,:,:]))[0,:,:]),torch.trace(cmul(cmul(U_new[:,i,:,:],W_new[:,i,:,:]),conjT(U_new[:,i,:,:]))[1,:,:])),I)
        temp2 +=  cmul(cmul(cmul(cmul(conjT(H[:,i,:,:]),U_new[:,i,:,:]),W_new[:,i,:,:]),conjT(U_new[:,i,:,:])),H[:,i,:,:])
    for i in range(K):
        B[:,i,:,:] = temp1 + temp2
    return B 

    
def E_update(K,d,U_new,V,H):
    E = torch.zeros((2,K,d,d))
    I1 = torch.eye(d)
    I2 = torch.zeros((d,d))
    I = mcat(I1,I2)
    for i in range(K):
        E[:,i,:,:] = I - cmul(cmul(conjT(U_new[:,i,:,:]),H[:,i,:,:]),V[:,i,:,:])
    return E

def Normalize_f(K,V,Pmax):
    temp1=0
    V_out = torch.zeros_like(V)
    for j in range(K):
        temp1 = temp1 + torch.trace(cmul(V[:,j,:,:],conjT(V[:,j,:,:]))[0,:,:])
    de = torch.sqrt(temp1)
    for i in range(K):
        V_out[:,i,:,:] = math.sqrt(Pmax)/de * V[:,i,:,:]   
    return V_out   
                                        

def FinalV(K,Nt,d,H,U_new,B_new,W_new):
    V_out = torch.zeros((2,K,Nt,d))
    for i in range(K):
        V_out[:,i,:,:] = cmul(cmul(cmul(cinv(B_new[:,i,:,:]),conjT(H[:,i,:,:])),U_new[:,i,:,:]),W_new[:,i,:,:])
    return V_out


class U(nn.Module):
    def __init__(self,K,Nr,d):
        super(U,self).__init__()
        self.K = K
        self.Nr = Nr
        self.d = d
        self.params1 = {}
        self.params2 = {}
        self.params3 = {}
        self.params4 = {}
        for i in range(K):
            self.params1["%d"%(i)] = nn.Parameter(torch.zeros(2,Nr,Nr))
            torch.nn.init.xavier_uniform_(self.params1["%d"%(i)],gain = 0.001)   #参数X
            self.params2["%d"%(i)] = nn.Parameter(torch.zeros(2,Nr,Nr))
            torch.nn.init.xavier_uniform_(self.params2["%d"%(i)],gain = 0.001)   #参数Y
            self.params3["%d"%(i)] = nn.Parameter(torch.zeros(2,Nr,Nr))
            torch.nn.init.xavier_uniform_(self.params3["%d"%(i)],gain = 0.01)   #参数Z
            self.params4["%d"%(i)] = nn.Parameter(torch.zeros(2,Nr,d))
            torch.nn.init.xavier_uniform_(self.params4["%d"%(i)],gain = 0.01)   #参数O
        self.params1 = nn.ParameterDict(self.params1)
        self.params2 = nn.ParameterDict(self.params2)
        self.params3 = nn.ParameterDict(self.params3)
        self.params4 = nn.ParameterDict(self.params4)
    def forward(self,H,V,A):
        U_out = torch.zeros((2,self.K,self.Nr,self.d))
        A_inv = torch.zeros_like(A)
        for i in range(self.K):
            A_inv[:,i,:,:] = cmul(self.params1["%d"%(i)],cdiv(A[:,i,:,:]))+self.params3["%d"%(i)]*0.01    
        for i in range(self.K):
            U_out[:,i,:,:] = cmul(cmul(A_inv[:,i,:,:],H[:,i,:,:]),V[:,i,:,:]) + self.params4["%d"%(i)]     
        return U_out
    
    
class W(nn.Module):
    def __init__(self,K,d):
        super(W,self).__init__()
        self.K = K
        self.d = d
        self.params1 = {}
        self.params2 = {}
        self.params3 = {}
        for i in range(K):
            self.params1["%d"%(i)] = nn.Parameter(torch.zeros(2,d,d))
            torch.nn.init.xavier_uniform_(self.params1["%d"%(i)],gain = 0.001)   #参数X
            self.params2["%d"%(i)] = nn.Parameter(torch.zeros(2,d,d))
            torch.nn.init.xavier_uniform_(self.params2["%d"%(i)],gain = 0.001)   #参数Y
            self.params3["%d"%(i)] = nn.Parameter(torch.zeros(2,d,d))
            torch.nn.init.xavier_uniform_(self.params3["%d"%(i)],gain = 0.01)   #参数Z  
        self.params1 = nn.ParameterDict(self.params1)
        self.params2 = nn.ParameterDict(self.params2)
        self.params3 = nn.ParameterDict(self.params3)
    
    def forward(self,E):
        W_out = torch.zeros((2,self.K,self.d,self.d))
        for i in range(self.K):
            W_out[:,i,:,:] = 0.1*cmul(self.params1["%d"%(i)],cdiv(E[:,i,:,:]))+self.params3["%d"%(i)]*0.01
        return W_out


class V(nn.Module):
    def __init__(self,K,Nt,d):
        super(V,self).__init__()
        self.K = K
        self.Nt = Nt
        self.d = d
        self.params1 = {}
        self.params2 = {}
        self.params3 = {}
        self.params4 = {}
        for i in range(K):
            self.params1["%d"%(i)] = nn.Parameter(torch.zeros(2,self.Nt,self.d))  #参数O
            torch.nn.init.xavier_uniform_(self.params1["%d"%(i)],gain = 0.001)
            self.params2["%d"%(i)] = nn.Parameter(torch.zeros(2,Nt,Nt))           #参数X
            torch.nn.init.xavier_uniform_(self.params2["%d"%(i)],gain = 0.01)
            self.params3["%d"%(i)] = nn.Parameter(torch.zeros(2,Nt,Nt))           #参数Y
            torch.nn.init.xavier_uniform_(self.params3["%d"%(i)],gain = 0.01)
            self.params4["%d"%(i)] = nn.Parameter(torch.zeros(2,Nt,Nt))           #参数Z
            torch.nn.init.xavier_normal_(self.params4["%d"%(i)],gain = 0.01)  
        self.params1 = nn.ParameterDict(self.params1)
        self.params2 = nn.ParameterDict(self.params2)
        self.params3 = nn.ParameterDict(self.params3)
        self.params4 = nn.ParameterDict(self.params4)   
        
    def forward(self,H,U,B,W):
        V_out = torch.zeros((2,self.K,self.Nt,self.d))
        B_inv = torch.zeros((2,self.K,self.Nt,self.Nt))
        for i in range(self.K):
            B_inv[:,i,:,:] = cmul(self.params2["%d"%(i)],cdiv(B[:,i,:,:]))+0.01*self.params4["%d"%(i)]
        
        for i in range(self.K):
            V_out[:,i,:,:] = cmul(cmul(cmul(B_inv[:,i,:,:],conjT(H[:,i,:,:])),U[:,i,:,:]),W[:,i,:,:]) + self.params1["%d"%(i)] 
        
        return V_out

        
        
    
  