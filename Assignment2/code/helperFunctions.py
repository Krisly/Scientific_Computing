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


def NewtonsMethodODE(FunJac, tk, xk, dt, xinit, tol, maxit, varargin):
    k = 0
    t = tk + dt
    x = xinit
    print("----1")
    print(x)
    print(t)
    [f,J] = FunJac(t,x,varargin)
    print(f)
    print(dt)
    R = x - f*dt - xk
    I = np.eye(np.size(xk))
    while (k < maxit) and (np.linalg.norm(R,np.inf) > tol):
        k += 1
        dRdx = I - J*dt
        dx = np.linalg.solve(dRdx,R)
        print("-"*4+"NMODE")
        print(x)
        print(dx)
        print(dRdx)
        print(R)
        x = x - dx
        [f,J] = FunJac(t,x,varargin)
        R = x - f*dt - xk
    return x


def ExplicitEulerFixedStepSize(fun,ta,tb,N,xa,kwargs):
    dt = (tb-ta)/N
    nx = np.size(xa,0)
    X  = np.zeros([N+1,nx])
    T  = np.zeros([N+1,1])
    T[0,:] = np.transpose(ta)
    X[0,:] = np.transpose(xa)
    for k in range(N):
        f = fun(T[k],X[k,:],kwargs)
        T[k+1,:] = T[k,:] + dt
        X[k+1,:] = X[k,:] + f*dt;
    return[T,X]


def ExplicitEulerAdaptiveStep(fun,tspan,x0,h0,abstol,reltol,kwargs):
    maxiter = 10
    epstol = 0.8
    facmin = 0.1
    facmax = 5.0
    
    t0 = tspan[0]
    tf= tspan[1]
    
    t = t0
    h = h0
    x = np.reshape(x0, (-1, 2))
    
    T = np.transpose(t)
    X = (x)
    
    while t < tf or maxiter > 0:
        maxiter = maxiter - 1
        if t+h>tf:
            h = tf-t
        f = fun(t,x[0],kwargs)
        
        AcceptStep = False
        
        while not AcceptStep:
            x1 = x + h*f
            
            hm = 0.5*h
            tm = t + hm
            xm = x + hm*f
            
            fm = fun(tm,xm[0],kwargs)
            
            x1hat = xm + hm*fm
            
            e = x1hat - x1
            
            
            r = max(max(abs(e)/np.maximum(abstol, abs(x1hat)*reltol)))
            #print(e)
            #print(r)
            AcceptStep = r <= 1
            #print(AcceptStep)
            if AcceptStep:
                t = t+h
                x = x1hat
                
                T = np.append( T, t)
                X = np.append( X, x ,axis=0 )
            h = np.maximum(facmin,np.minimum(np.sqrt(epstol/r),facmax))*h
    return[T,X]


def ImplicitEulerFixedStepSize(funJac,ta,tb,N,xa,kwargs):
    
    # Compute step size and allocate memory
    dt = (tb-ta)/N
    nx = np.size(xa,0)
    X  = np.zeros([N+1,nx])
    T  = np.zeros([N+1,1])
    
    tol = 1.0e-8
    maxit = 100
    
    #Eulers Implicit Method
    T[0,:] = np.transpose(ta)
    X[0,:] = np.transpose(xa)
    for k in range(N):
        [f,j] = funJac(T[k],X[k,:],kwargs)
        T[k+1,:] = T[k,:] + dt
        xinit = X[k,:] + np.transpose(f*dt)[0]
        X[k+1,:] = NewtonsMethodODE(funJac,
                                    T[k,:],
                                    X[k,:],
                                    dt,
                                    xinit,
                                    tol,
                                    maxit,
                                    kwargs);

    T = np.transpose(T)
    X = np.transpose(X)
    return [T,X]