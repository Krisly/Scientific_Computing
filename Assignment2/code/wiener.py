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
from numsolver import numerical_solvers

ex = numerical_solvers()


def ScalarStdWienerProcess(T,N,Ns,seed):

    np.random.seed(seed)
    dt = T/N
    dW = np.sqrt(dt)*np.random.randn(Ns,N)
    W = np.append(np.zeros([Ns,1]), np.cumsum(dW,1), axis=1)
    Tw = np.arange(0,T+dt/2,dt)
    return [W,Tw,dW]


def StdWienerProcess(T,N,nW,Ns,seed):

    np.random.seed(seed)
    dt = T/N
    dW = np.sqrt(dt)*np.random.randn(Ns,nW,N)
    W = np.append(np.zeros([Ns,1,nW]), np.cumsum(dW,1), axis=2)
    Tw = np.arange(0,T+dt/2,dt)
    return [W,Tw,dW]


def SDEeulerExplicitExplicit(ffun,gfun,T,x0,W,varargin):
    N = np.size(T)-1
    nx = np.size(x0)
    X = np.zeros([nx,N+1])
    X[:,0] = x0
    for k in range(0,N):
        dt = T[k+1]-T[k]
        dW = W[k+1]-W[k]
        f = ffun(T[k],X[:,k],varargin)
        g = gfun(T[k],X[:,k],varargin)
        X[:,k+1] = X[:,k] + f*dt + g*dW
    return X

def SDENewtonSolver(ffun,t,dt,psi,xinit,tol,maxit,varargin):
    I = np.eye(np.size(xinit))
    x = xinit
    [f,J] = ffun(t,x,varargin)
    R = x - f*dt - psi
    it = 1
    while ( (np.norm(R,np.inf) > tol) and (it <= maxit) ):
        dRdx = I - J*dt
        mdx = np.linalg.solve(dRdx,R)
        x = x - mdx
        [f,J] = ffun(t,x,varargin)
        R = x - f*dt - psi
        it = it+1
    return [x,f,J]

def SDEeulerImplicitExplicit(ffun,gfun,T,x0,dW,varargin):
    N = np.size(T)-1
    nx = np.size(x0)
    X = np.zeros([nx,N+1])
    X[:,0] = x0
    for k in range(0,N):
        dt = T[k+1]-T[k]
        f = ffun(T[k],X[:,k],varargin)
        g = gfun(T[k],X[:,k],varargin)
        psi = X[:,k] + g*dW[k]
        X = psi + f*dt
        X[:,k+1] = SDENewtonSolver(ffun,T[k+1],dt,psi,x0,1.0e-8,100,varargin)
    return X
#    tol = 1.0e-8
#    maxit = 100
#    N = np.size(T)
#    nx = np.size(x0)
#    X = np.zeros([nx,N])
#    X[:,0] = x0
#    k=1
#    f = ffun(T[k],X[:,k],varargin)
#    for k in range(0, N-1):
#        g = gfun(T[k],X[:,k],varargin)
#        dt = T[k+1]-T[k]
#        dW = W[:,k+1]-W[:,k]
#        psi = X[:,k] + g*dW
#        xinit = psi + f*dt
#        [X[:,k+1],f,qq] = SDENewtonSolver(ffun,T[k+1],dt,psi,xinit,tol,maxit,varargin)
#    return X
def VanderpolDrift(t,x,p):
    mu = p[0]
    tmp = mu*(1.0-x[0]*x[0])
    f = np.zeros([2,1])
    f[0,0] = x[1]
    f[1,0] = tmp*x[1]-x[0]
    J = np.array([[0, 1],[ -2*mu*x[0]*x[1]-1.0, tmp]])
    return [f,J]

def VanderPolDiffusion1(t,x,p):
    sigma = p[1]
    g = [0.0, sigma]
    return g

def LangevinDrift(t,x,p):
    lamda = p[0]
    f = lamda*x
    return f

def LangevinDiffusion(t,x,p):
    sigma = p[1]
    g = sigma
    return g

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

P = [lamda, sigma]

[W,Tw,dW]=ScalarStdWienerProcess(T,N,Ns,seed)

X = np.zeros([np.size(W,0), np.size(W,1)])
for i in range(0,Ns):
    X[i,:] = SDEeulerExplicitExplicit(LangevinDrift,LangevinDiffusion,Tw,x0,W[i,:],P)

plt.plot(Tw,X.T)
plt.show()

#
#
#[W,Tw,dW]=ScalarStdWienerProcess(T,N,Ns,seed)
#
#X = np.zeros([np.size(W,0), np.size(W,1)])
#for i in range(0,Ns):
#    X[i,:] = SDEeulerImplicitExplicit(LangevinDrift,LangevinDiffusion,Tw,x0,W[i,:],P)
#
#plt.plot(Tw,X.T)
#plt.show()
#

mu = 3
sigma = 1
x0 = [0.5, 0.5]
p = [mu, sigma]
tf = 5*mu
nw = 1
N = 1000
Ns = 5
seed = 100

[W,T,dW]=StdWienerProcess(tf,N,nw,Ns,seed)
X = np.zeros([np.size(W,0), np.size(W,1)])
for i in range(0,Ns):
    X[i,:] = SDEeulerExplicitExplicit(VanderpolDrift,VanderPolDiffusion1,T,x0,W[i,:],P)
#X(:,:,i) = SDEsolverExplicitExplicit(...
#@VanderPolDrift,@VanderPolDiffusion1,...
#T,x0,W(:,:,i),p);
