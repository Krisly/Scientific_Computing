#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:36:20 2018

@author: carl
"""
import numpy as np


def NewtonsMethod(ResidualFunJac, x0, tol, maxit, varargin):
    # General Newton's method
    k = 0
    x = x0
    [R,dRdx] = ResidualFunJac(x,varargin)
    while (k < maxit) and (np.linalg.norm(R,np.inf) > tol):
        k += 1
        dx = np.linalg.solve(dRdx,R)
        x = x - dx
        [R,dRdx] = ResidualFunJac(x,varargin)
    return x


def NewtonsMethodODE(FunJac, tk, xk, dt, xinit, tol, maxit, varargin):
    # Newton's method for single stage RK schemes (Euler's)
    k = 0
    t = tk + dt
    x = xinit

    [f,J] = FunJac(t,x,varargin)

    R = x - f*dt - xk
    I = np.eye(np.size(xk))
    while (k < maxit) and (np.linalg.norm(R,np.inf) > tol):
        k += 1
        dRdx = I - J*dt
        dx = np.linalg.solve(dRdx,R)

        x = x - dx
        [f,J] = FunJac(t,x,varargin)
        R = x - f*dt - xk
    return x


def ExplicitEulerFixedStepSize(fun,ta,tb,N,xa,kwargs):
    # Explicit Euler's, fixed step size
    dt = (tb-ta)/N
    nx = np.size(xa)
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
    # Explicit Euler's, adaptive step size (infinity norm)
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

            AcceptStep = r <= 1

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

def ExplicitRungeKuttaSolver(fun,tspan,x0,h,solver,varargin):
    #% EXPLICITRUNGEKUTTASOLVER  Fixed step size ERK solver for systems of ODEs
    #%
    #% Syntax:
    #% [Tout,Xout]=ExplicitRungeKuttaSolver(fun,tspan,x0,h,solver,varargin)
    #% Solver Parameters
    
    s  = solver.stages     #% Number of stages in ERK method
    AT = solver.AT         #% Transpose of A-matrix in Butcher tableau
    b  = solver.b          #% b-vector in Butcher tableau
    c  = solver.c          #% c-vector in Butcher tableau
    
    # Parameters related to constant step size
    hAT = h*AT
    hb  = h*b
    hc  = h*c
    # Size parameters
    x  = x0
    t  = tspan[0]          #% Initial time
    tf = tspan[-1]        #% Final time
    N = (tf-t)/h           #% Number of steps
    nx = np.size(x0)        #% System size (dim(x))
    #% Allocate memory
    T  = np.zeros(1,s)        #% Stage T
    X  = np.zeros(nx,s)       #% Stage X
    F  = np.zeros(nx,s)       #% Stage F
    Tout = np.zeros(N+1,1)    #% Time for output
    Xout = np.zeros(N+1,nx)   #% States for output
    
    Tout[0] = t
    Xout[0,:] = x.T
    for n in range(0,N):
    #% Stage 1
        T[0]   = t
        X[:,0] = x
        F[:,0] = fun(T[0],X[:,0],varargin)
        #% Stage 2,3,...,s
        T[1:s] = t + hc[1:s]
        for i in range(1,s):
            X[:,i] = x + F[:,0:i-1]*hAT[0:i-1,i]
            F[:,i] = fun(T[i],X[:,i])
        #% Next step
        t = t + h
        x = x + F*hb
        #% Save output
        Tout[n+1] = t
        Xout[n+1,:] = x.T
    
    return [Tout,Xout]