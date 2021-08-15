import torch
import numpy as np
import matplotlib.pyplot as plt

def WMMSE(H_tensor, PT, sigma):
    K = H_tensor.size(2) #用户数

    Hnp = H_tensor.numpy()
    Hre = Hnp[0, :]
    Him = Hnp[1, :]
    Hc = Hre + 1j * Him

    Nt = H_tensor.size(1) #发送天线
    Nr = 1 #接收天线

    MaxIter = 200 #最大迭代次数
    epsilon = 1e-6 #收敛精度

    H = []
    for i in range(K):
        # H.append(np.sqrt(1/2)*(np.random.randn(Nr,Nt)+1j*np.random.randn(Nr,Nt)))

        H.append(np.expand_dims(Hc[:,i].T,axis=0))

    V = []
    for i in range(K):
        V.append(np.sqrt(1/2)*(np.random.randn(Nt,Nr)+1j*np.random.randn(Nt,Nr)))


    #初始化：k个Tr(Vk*Vk^T)之和 = PT
    alpha = np.sqrt(PT/np.sum([np.trace(np.dot(V[i],V[i].conjugate().T)) for i in range(K)]))
    for k in range(K):
        V[k] = alpha * V[k]
    Tr_V = np.sum([np.trace(np.dot(V[i],V[i].conjugate().T)) for i in range(K)])


    #First iteration
    A = []
    U = []
    for k in range(K):
        A_item = np.zeros([Nr,Nr]) + 0j*np.zeros([Nr,Nr])
        for m in range(K):
            A_item += H[k].dot(V[m]).dot(V[m].conjugate().T).dot(H[k].conjugate().T)
        A.append(sigma**2/PT*np.sum([np.trace(np.dot(V[i],V[i].conjugate().T)) for i in range(K)])*np.eye(Nr) + A_item)
        U.append(np.linalg.pinv(A[k]).dot(H[k]).dot(V[k]))

    W = []
    W_old = []
    for k in range(K):
        E = np.eye(Nr) + 0j*np.eye(Nr) -(U[k].conjugate().T).dot(H[k]).dot(V[k])
        W.append(np.linalg.pinv(E))
        W_old.append(W[k])

    V = []
    for k in range(K):
        B_1 = np.zeros([Nt,Nt]) + 0j*np.zeros([Nt,Nt])
        for m in range(K):
            B_1 += (H[m].conjugate().T).dot(U[m]).dot(W[m]).dot(U[m].conjugate().T).dot(H[m])
        B = B_1 + sigma**2/PT*np.sum([np.trace(U[i].dot(W[i]).dot(U[i].conjugate().T)) for i in range(K)])*np.eye(Nt)
        V.append(np.linalg.pinv(B).dot(H[k].conjugate().T).dot(U[k]).dot(W[k]))

    #Scale
    alpha = np.sqrt(PT/np.sum([np.trace(np.dot(V[i],V[i].conjugate().T)) for i in range(K)]))
    for k in range(K):
        V[k] = alpha * V[k]


    iteration = []
    rate = []
    t = 0
    while(t<MaxIter):
        #Update of Uk
        for k in range(K):
            A_item = np.zeros([Nr,Nr]) + 0j*np.zeros([Nr,Nr])
            for m in range(K):
                A_item += H[k].dot(V[m]).dot(V[m].conjugate().T).dot(H[k].conjugate().T)
            A[k] = sigma**2/PT*np.sum([np.trace(np.dot(V[i],V[i].conjugate().T)) for i in range(K)])*np.eye(Nr) + A_item
            U[k] = np.linalg.pinv(A[k]).dot(H[k]).dot(V[k])

        #Update of Wk
        for k in range(K):
            E = np.eye(Nr) + 0j*np.eye(Nr) -(U[k].conjugate().T).dot(H[k]).dot(V[k])
            W_old[k] = W[k]
            W[k] = np.linalg.pinv(E)


        #Update of Vk
        for k in range(K):
            B_1 = np.zeros([Nt,Nt]) + 0j*np.zeros([Nt,Nt])
            for m in range(K):
                B_1 += (H[m].conjugate().T).dot(U[m]).dot(W[m]).dot(U[m].conjugate().T).dot(H[m])
            B = B_1 + sigma**2/PT*np.sum([np.trace(U[i].dot(W[i]).dot(U[i].conjugate().T)) for i in range(K)])*np.eye(Nt)
            V[k] = np.linalg.pinv(B).dot(H[k].conjugate().T).dot(U[k]).dot(W[k])

        #Scale
        alpha = np.sqrt(PT/np.sum([np.trace(np.dot(V[i],V[i].conjugate().T)) for i in range(K)]))
        for k in range(K):
            V[k] = alpha * V[k]

        rate.append(np.sum(np.log2(np.linalg.det(W[k])) for k in range(K)))
        t += 1;
        iteration.append(t)
#print(abs(np.sum([np.log(np.linalg.det(W[i])) for i in range(K)]) - np.sum([np.log(np.linalg.det(W_old[i])) for i in range(K)])))


    # plt.plot(iteration,rate)
    # plt.xlabel("iteration")
    # plt.ylabel("rate")
    # plt.show()

    return rate[t-1].real