#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:27:26 2018

@author: carl
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import helperFunctions as hF



def ScalarStdWienerProcess(T,N,Ns,seed):

    np.random.seed(seed)
    dt = T/N
    dW = np.sqrt(dt)*np.random.randn(Ns,N)
    W = np.append(np.zeros([Ns,1]), np.cumsum(dW,1), axis=1)
    Tw = np.arange(0,T+dt/2,dt)
    return [W,Tw,dW]

[W,Tw,dW] = ScalarStdWienerProcess(20,2000,20,100)
#np.tile(Tw,[np.size(W,0),1])
plt.plot(Tw,W.T)
plt.show()



T = 10
N = 10000
Ns = 10
seed = 1002


x0 = 10
lamda = -0.5
sigma = 1.0
[W,Tw,dW]=ScalarStdWienerProcess(T,N,Ns,seed)
X = np.zeros([np.size(W,0), np.size(W,1)])

for i in range(0,Ns):
    X[i,0] = x0;
    for k in range(0,N):
        dt = Tw[k+1]-Tw[k];
        X[i,k+1] = X[i,k] + lamda*X[i,k]*dt + sigma*dW[i,k]*X[i,k];

plt.plot(Tw,X.T)
plt.show()