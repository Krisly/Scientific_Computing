#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:36:20 2018

@author: carl
"""
import numpy as np

def NewtonsMethod(ResidualFunJac, x0, tol, maxit, varargin):
    k = 0
    x = x0
    [R,dRdx] = ResidualFunJac(x,varargin)
    print(R, dRdx)
    while (k < maxit) and (np.linalg.norm(R,np.inf) > tol):
        k += 1
        dx = np.linalg.solve(dRdx,R)
        x = x - dx
        [R,dRdx] = ResidualFunJac(x,varargin)
    return x


def NewtonsMethodODE(FunJac, tk, xk, dt, xinit, tol, maxit, kwargs):
    k = 0
    t = tk + dt
    x = xinit
    [f,J] = FunJac(t,x,kwargs)
    R = x - f*dt - xk
    I = np.eye(np.size(xk))
    while (k < maxit) and (np.linalg.norm(R,np.inf) > tol):
        k += 1
        dRdx = I - J*dt
        dx = np.linalg.solve(dRdx,R)
        x = x - dx
        [f,J] = FunJac(t,x,kwargs)
        R = x - f*dt - xk
    return x

'''
def ImplicitEulerFixedStepSize(funJac, ta, tb, N, xa, varargin):
    dt = (tb-ta)/N
    nx = np.size(xa,1)
    X = np.zeros([nx,N+1])
    T = np.zeros([1,N+1])
    tol = 1.0e-8
    maxit = 100
    T[:,1] = ta
    X[:,1] = xa
    for k in range(1, N):
        f = fun(T[k],X[:,k],varargin)
        T[:,k+1] = T[:,k] + dt
        xinit = X[:,k] + f*dt
        X[:,k+1] = NewtonsMethodODE(funJac, T[:,k], X[:,k], dt, xinit, tol, maxit, varargin)
    T = np.transpose(T)
    X = np.transpose(X)
    return [T,X]
    '''
    
def ImplicitEulerFixedStepSize(funJac,ta,tb,N,xa,kwargs=None):

    # Compute step size and allocate memory
    dt = (tb-ta)/N;
    nx = np.size(xa);
    X  = np.zeros([N+1,nx]);
    T  = np.zeros([N+1,1]);
    
    tol = 10**(-8);
    maxit = 100;

    #Eulers Implicit Method
    T[0,:] = np.transpose(ta);
    X[0,:] = np.transpose(xa);
    for k in range(N-1):
        [f,j] = funJac(T[k],X[k,:],kwargs);
        T[k+1,:] = T[k,:] + dt;
        xinit = X[k,:] + np.transpose(f*dt);
        X[k+1,:] = NewtonsMethodODE(funJac,
                                    T[k,:],
                                    X[k,:],
                                    dt,
                                    xinit,
                                    tol,
                                    maxit,kwargs);
         
    return T,X