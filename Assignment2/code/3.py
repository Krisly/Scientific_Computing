#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 20:13:39 2018

@author: carl
"""

import numpy as np
import helperFunctions as hF
import matplotlib.pyplot as plt

def testEqn(t,x,p):
    # function for the test problem
    lamda = p[0]
    return lamda*x

def JacTestEqn(t,x,p):
    # jacobian for the test problem, in this case simply derivative
    return p[0]

def testEqnFunJac(t,x,p):
    # combinatory function
    return testEqn(t,x,p), JacTestEqn(t,x,p)

def testEqnAnalyt(t,x0,p):
    # analytical solution for the test problem
    lamda = p[0]
    return x0*np.exp(lamda*t)

lamda = -1

ta = 0
tb = 10
N = 100

xa = 1
kwargs = [lamda]

LERR = []
GERR = []
dt = []

fig, ax = plt.subplots(1, 2, figsize=(10,5), sharex=False)

for N in range(100,1000,10):

    dt = np.append(dt, tb/N)
    
    # Numerical estimate
    Te,Xe = hF.ExplicitEulerFixedStepSize(testEqn,ta,tb,N,xa,kwargs)
    
    # Analytical solution
    Tanal = np.linspace(ta,tb,N+1)
    Xanal = testEqnAnalyt(Tanal,xa,kwargs)
    
    # Local Error
    LERR = np.append(LERR, np.abs(Xanal[1] - Xe[1]))
    # Global Error
    GERR = np.append(GERR, np.max( np.abs(Xanal - Xe.T)) )

# ------------------------- 1-3-4
ax[0].loglog(dt,LERR, label='Local Error')
ax[0].loglog(dt,GERR, label='Global Error')
ax[0].loglog(dt,dt**1, label='O(h^1)')
ax[0].loglog(dt,dt**2, label='O(h^2)')
ax[0].legend() 
ax[0].set_title('Orders for the Explicit Euler')
ax[0].set_xlabel('dt')
ax[0].set_ylabel('E')

LERR = []
GERR = []
dt = []

for N in range(100,1000,10):

    dt = np.append(dt, tb/N)
    
    # Numerical estimate
    Ti,Xi = hF.ImplicitEulerFixedStepSize(testEqnFunJac,ta,tb,N,[xa],kwargs)
    Xi = Xi[0]
    
    # Analytical solution
    Tanal = np.linspace(ta,tb,N+1)
    Xanal = testEqnAnalyt(Tanal,xa,kwargs)
    
    # Local Error
    LERR = np.append(LERR, np.abs(Xanal[1] - Xi[1]))
    # Global Error
    GERR = np.append(GERR, np.max(np.abs(Xanal - Xi)) )

# ------------------------- 1-3-4
ax[1].loglog(dt,LERR, label='Local Error')
ax[1].loglog(dt,GERR, label='Global Error')
ax[1].loglog(dt,dt**1, label='O(h^1)')
ax[1].loglog(dt,dt**2, label='O(h^2)')
ax[1].legend()
ax[1].set_title('Orders for the Implicit Euler')
ax[1].set_xlabel('dt')
ax[1].set_yticks([])


fig.savefig('3-2.eps', format='eps', dpi=500, bbox_inches='tight')

# ------------------------- 1-3
fig, ax = plt.subplots(1, 2, figsize=(10,5), sharex=False)
ax[0].plot(Te,Xe, label='Explicit Euler')
ax[0].plot(Tanal,Xanal, '--', label='Analytical')
ax[0].legend() 
ax[0].set_title('Explicit Euler')
ax[0].set_xlabel('t')
ax[0].set_ylabel('x')

# ------------------------- 1-3
ax[1].plot(Ti[0],Xi, label='Implicit Euler')
ax[1].plot(Tanal,Xanal, '--',label='Analytical')
ax[1].legend()
ax[1].set_title('Implicit Euler')
ax[1].set_xlabel('t')
ax[1].set_ylabel('x')

fig.savefig('3-2-2.eps', format='eps', dpi=500, bbox_inches='tight')