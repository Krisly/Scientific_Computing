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
    dW = np.sqrt(dt)*np.random.randn(nW,N,Ns)
    W = np.append(np.zeros([nW,1,Ns]), np.cumsum(dW,1), axis=1)
    Tw = np.arange(0,T+dt/2,dt)
    return [W,Tw,dW]


def SDEeulerExplicitExplicit(ffun,gfun,T,x0,W,varargin):
    N = np.size(T)-1
    nx = np.size(x0)
    X = np.zeros([nx,N+1])
    X[:,0] = x0
    for k in range(0,N):
        dt = T[k+1]-T[k]
        dW = W[k+1,:]-W[k,:]
        f = ffun(T[k],X[:,k],varargin)
        g = gfun(T[k],X[:,k],varargin)
        X[:,k+1] = X[:,k] + f*dt + g*dW
    return X

def SDENewtonSolver(ffun,t,dt,psi,xinit,tol,maxit,varargin):
    I = np.eye(np.size(xinit))
    x = xinit
    [f,J] = ffun(t,x,varargin)
    R = x - f[0]*dt - psi
    it = 1
    while ( (np.linalg.norm(R,np.inf) > tol) and (it <= maxit) ):
        dRdx = I - J*dt
        mdx = np.linalg.solve(dRdx,R)
        x = x - mdx
        [f,J] = ffun(t,x,varargin)
        R = x - f[0]*dt - psi
        it = it+1
    return [x,f,J]

def SDEeulerImplicitExplicit(ffun,gfun,T,x0,W,varargin):
    tol = 1e-8
    maxit = 100
    
    N = np.size(T)-1
    nx = np.size(x0)
    X = np.zeros([nx,N+1])
    X[:,0] = x0
    k = 0
    [f,j] = ffun(T[k],X[:,k],varargin)
    
    for k in range(0,N):
        dt = T[k+1]-T[k]
        g = gfun(T[k],X[:,k],varargin)
        dW = W[k+1,:]-W[k,:]
        psi = X[:,k] + g*dW
        xinit = psi + f[0]*dt
        X[:,k+1],f,j = SDENewtonSolver(ffun,T[k+1],dt,psi,xinit,tol,maxit,varargin)
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
    f = np.zeros([1,2])
    f[0,0] = x[1]
    f[0,1] = tmp*x[1]-x[0]
    J = np.array([[0, 1],[ -2*mu*x[0]*x[1]-1.0, tmp]])
    return [f,J]

def VanderPolDiffusion1(t,x,p):
    sigma = p[1]
    g = [0.0, sigma]
    return g

def VanderPolDiffusion2(t,x,p):
    sigma = p[1]
    g = [0.0, sigma*(1+x[0]*x[1])]
    return g

def LangevinDrift(t,x,p):
    lamda = p[0]
    f = lamda*x
    return f

def LangevinDiffusion(t,x,p):
    sigma = p[1]
    g = sigma
    return g

def GeometricBrownianDrift(t,x,p):
    lamda = p[0]
    f = lamda*x
    return f

def GeometricBrownianDiffusion(t,x,p):
    sigma = p[1]
    g = sigma*x
    return g

[W,Tw,dW] = ScalarStdWienerProcess(20,2000,20,100)
#np.tile(Tw,[np.size(W,0),1])
plt.plot(Tw,W.T)
plt.show()



T = 10
N = 1000
Ns = 100
seed = 1001


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
    X[i,:] = SDEeulerExplicitExplicit(LangevinDrift,LangevinDiffusion,Tw,x0,W[i,:,None],P)

plt.plot(Tw,X.T)
plt.show()

x0 = 1
lamda = 0.1
sigma = 0.15
Ns = 10000
[W,Tw,dW]=ScalarStdWienerProcess(T,N,Ns,seed)

P = [lamda, sigma]

X = np.zeros([np.size(W,0), np.size(W,1)])

for i in range(0,Ns):
    X[i,:] = SDEeulerExplicitExplicit(GeometricBrownianDrift,GeometricBrownianDiffusion,Tw,x0,W[i,:,None],P)

#x0*exp((mu-s**2/2)*t+sw)
#np.exp(W[1,:])

#x0*np.exp( (lamda-sigma**2/2)*Tw + sigma*W[4,:])

mean = np.mean(X.T,axis=1)
meanW = np.mean(W,axis=0)

plt.figure()
plt.plot(Tw,x0*np.exp( (lamda-0**2/2)*Tw + 0*W[2,:]),'g')
plt.plot(Tw,mean, '-')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend(['analytical, sigma=0','mean of Euler–Maruyama'])
plt.show()

plt.plot(Tw,X.T)
plt.plot(Tw,mean,'black',linewidth=3)
plt.plot(Tw,np.std(X.T,axis=1)+mean,'black',linewidth=3)
plt.plot(Tw,-np.std(X.T,axis=1)+mean,'black',linewidth=3)
plt.xlim([Tw[0],Tw[-1]])
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()

plt.subplot(121)
plt.plot(Tw,x0*np.exp( (lamda-sigma**2/2)*Tw + sigma*W[2,:]) ,'g',Tw,X.T[:,2],'-')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend(['analytical','Euler–Maruyama'])
plt.subplot(122)
plt.plot(Tw,x0*np.exp( (lamda-sigma**2/2)*Tw + sigma*W[2,:]) ,'g',Tw,X.T[:,2],'-')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend(['analytical','Euler–Maruyama'])
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
P = [mu, sigma]
tf = 5*mu
Nw = 2
N = 1000
Ns = 15
seed = 1002

[W,T,dW]=StdWienerProcess(tf,N,Nw,Ns,seed)

plt.plot(T,W[1,:,:])
plt.title(str(Ns) + ' realizations of the standard Wiener process')
plt.xlabel('t')
plt.ylabel('W(t)')
plt.xlim([T[0],T[-1]])
plt.show()

X = np.zeros([np.size(x0), N+1, Ns])
Ximpl = np.zeros([np.size(x0), N+1, Ns])

for i in range(0,Ns):
    X[:,:,i] = SDEeulerExplicitExplicit(lambda t,x,p: VanderpolDrift(t,x,p)[0],VanderPolDiffusion1,T,x0,W[:,:,i].T,P)
    Ximpl[:,:,i] = SDEeulerImplicitExplicit(VanderpolDrift, VanderPolDiffusion1, T, x0, W[:,:,i].T, P)



r = ode(lambda t,x,p: VanderpolDrift(t,x,p)[0],
        lambda t,x,p: VanderpolDrift(t,x,p)[1]).set_integrator('vode',
                                      method='bdf',
                                      with_jacobian=True,
                                      order=15)
r.set_initial_value(x0, 0).set_f_params(P).set_jac_params(P)
x_sci_s = [[],[]]
t       = np.arange(0,tf+0.01,0.01)
# Solving using the scipy solver
while r.successful() and r.t < tf:
    xn = r.integrate(r.t+0.01)
    x_sci_s[0].append(xn[0])
    x_sci_s[1].append(xn[1])

plt.figure()
#plt.title('Solution for the stochastic VDP problem wuth mu=' +str(mu)+ ', sigma='+str(sigma))
plt.subplot(131)
plt.plot(T, X[0,:,:])
plt.plot(t, x_sci_s[0], 'black', linewidth=3)
plt.xlabel('t')
plt.ylabel('X1')
plt.subplot(132)
plt.plot(T, X[1,:,:])
plt.plot(t, x_sci_s[1], 'black', linewidth=3)
plt.xlabel('t')
plt.ylabel('X2')
plt.subplot(133)
plt.plot(X[0,:,:],X[1,:,:])
plt.plot(x_sci_s[0],x_sci_s[1], 'black', linewidth=3)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

plt.figure()
#plt.title('Solution for the stochastic VDP problem wuth mu=' +str(mu)+ ', sigma='+str(sigma))
plt.subplot(131)
plt.plot(T, Ximpl[0,:,:])
plt.plot(t, x_sci_s[0], 'black', linewidth=3)
plt.xlabel('t')
plt.ylabel('X1')
plt.subplot(132)
plt.plot(T, Ximpl[1,:,:])
plt.plot(t, x_sci_s[1], 'black', linewidth=3)
plt.xlabel('t')
plt.ylabel('X2')
plt.subplot(133)
plt.plot(Ximpl[0,:,:],Ximpl[1,:,:])
plt.plot(x_sci_s[0],x_sci_s[1], 'black', linewidth=3)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

#plt.plot(t, x_sci_s[0], t ,x_sci_s[1])
#plt.show()
#plt.plot(x_sci_s[0],x_sci_s[1])
#plt.show()

#X(:,:,i) = SDEsolverExplicitExplicit(...
#@VanderPolDrift,@VanderPolDiffusion1,...
#T,x0,W(:,:,i),p);
