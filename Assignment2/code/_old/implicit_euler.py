# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def ImplicitEulerFixedStepSize(funJac,ta,tb,N,xa,**kwargs):

    # Compute step size and allocate memory
    dt = (tb-ta)/N;
    nx = size(xa,0);
    X  = np.zeros(N+1,nx);
    T  = np.zeros(N+1,1);
    
    tol = 1.0e-8;
    maxit = 100;

    #Eulers Implicit Method
    T[0,:] = ta;
    X[0,:] = xa;
    for k in range(N):
        [f,j] = funJac(T[k],X[k,:],**kwargs);
        T[k+1,:] = T[k,:] + dt;
        xinit = X[k,:] + f*dt;
        X[k+1,:] = NewtonsMethodODE(funJac,
                                    T[k,:],
                                    X[k,:],
                                    dt,
                                    xinit,
                                    tol,
                                    maxit,
                                    **kwargs);

    T = np.transpose(T)
    X = np.transpose(X)