import numpy as np
import math
from matplotlib import pyplot as plt

def beamspace_channel(M, K, L, angle=0):
    """M 天线数   K 用户数   L 路径数"""

    m = np.arange(M)
    fai_m = 1 / M * (m - (M ) / 2);

    '''傅里叶变换矩阵'''
    U = []
    for i in range(M):
        U.append(1/math.sqrt(M)*np.exp(-1j*2*np.pi*fai_m[i]*m))

    U = np.array(U)
    U = U.conjugate()

    G = []
    for k in range(K):
        #Los_gain = (1+np.random.randn(1)*0.5)*np.exp(1j*np.random.rand(1)*2*np.pi) 
        #距离默认是100  pathloss增加了10^10 不然太小
        # Los_loss = 61.4 + 20*2 + np.random.randn(1)*math.pow(10,0.58)
        # Los_loss = math.pow(10, (100-Los_loss)/10)
        # Los_gain = Los_gain*Los_loss
        Los_gain = np.random.randn(1) + 1j*np.random.randn(1)
        #Los_gain = (1 + np.random.randn(1) * 0.2) * np.exp(1j * np.random.rand(1) * 2 * np.pi)
        theta =  2*np.pi * np.random.rand(1)+np.pi/2
        #固定角度
        #theta = np.pi * k / K+np.pi/2#+np.random.randn(1)*1/K*np.pi+angle
        fai = np.sin(theta)/2
        fai = (np.random.rand(1)-0.5)#*1/K +k/K
        g =  Los_gain*1/math.sqrt(M)*np.exp(-1j*2*np.pi*fai*m);
        for l in range(L-1):
            # nLos_gain = np.random.randn(1) + 1j * np.random.randn(1)
            # nLos_loss = 72 + 29.2 * 2 + np.random.randn(1) * math.pow(10, 0.87)
            # nLos_loss = math.pow(10, (100 - nLos_loss) / 10)
            # nLos_gain = nLos_gain * nLos_loss
            nLos_gain = np.sqrt(0.2)*(np.random.randn(1) + 1j*np.random.randn(1))
            theta = 2 * np.pi * np.random.rand(1)
            fai = np.sin(theta) / 2
            fai = np.random.rand(1)-0.5
            g = g + nLos_gain * 1 / math.sqrt(M) * np.exp(-1j * 2 * np.pi * fai * m);
        G.append(g)
    G = np.array(G)
    H = np.dot(U,G.T)


    """画出每个beam的能量"""
    beam_energy = np.sum(np.abs(H),1)
    # print(beam_energy)
    # plt.plot(m,beam_energy)
    # plt.show()

    return  H