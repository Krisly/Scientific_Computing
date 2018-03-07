#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:29:32 2018

@author: root
"""
from numsolver import numerical_solvers
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

# Sets Plotting parameters
font_size = 15
plt.rcParams['figure.figsize'] = (15,7)
plt.rc('font',   size=font_size)       # controls default text sizes
plt.rc('axes',  titlesize=font_size)   # fontsize of the axes title
plt.rc('axes',   labelsize=font_size)  # fontsize of the x any y labels
plt.rc('xtick',  labelsize=font_size)  # fontsize of the tick labels
plt.rc('ytick',  labelsize=font_size)  # fontsize of the tick labels
plt.rc('legend', fontsize=font_size)   # legend fontsize
plt.rc('figure', titlesize=font_size)  # # size of the figure title

# Setting up solver parameters
mu              = 5
x0              = np.array([2.0,0.0])
tend            = 20
numsolv1        = numerical_solvers()
numsolv1.param  = mu
numsolv1.x0     = x0
numsolv1.maxit  = 100
numsolv1.t      = np.array([0,tend])
numsolv1.dt     = 0.01
t               = np.arange(0,tend+0.01,0.01)
numsolv1.absTol = 10**(-8)
numsolv1.relTol = 10**(-3)
numsolv1.epstol = 0.8
absTol_as       = 10**(-4) 

# Solving using the Implicit fixed step size Euler method
sol_T,sol_X = numsolv1.ImplicitEulerFixedStepSize(numsolv1.VanderPolfunjac)

# Solving using the adaptive Implicit Euler
a_sol_T,a_sol_X,a_ss,reject = numsolv1.ImplicitEulerAdaptiveStepSize(
                                                    numsolv1.VanderPolfunjac,
                                                    absTol_as)
# Setting up scipy solver
r = ode(numsolv1.VanDerPol,
        numsolv1.JacVanDerPol).set_integrator('vode',
                                              method='bdf',
                                              order=15)
r.set_initial_value(x0, 0).set_f_params(mu).set_jac_params(mu)
x = [[],[]]

# Solving using the scipy solver
while r.successful() and r.t < tend:
    xn = r.integrate(r.t+0.01)
    x[0].append(xn[0])
    x[1].append(xn[1])

# Plotting the results
plt.plot(t[:len(x[0])], x[0],label='Scipy solver')
plt.plot(sol_T, sol_X[:,0],label='Implicit Euler fixed step')
plt.plot(a_sol_T, a_sol_X[:,0],label='Implicit Euler adaptive step')
plt.legend(loc='lower left')
plt.figure()
plt.plot(t[:len(x[1])], x[1],label='Scipy Solver')
plt.plot(sol_T, sol_X[:,1],label='Implicit Euler')
plt.plot(a_sol_T, a_sol_X[:,1],label='Implicit Euler adaptive step')
plt.legend(loc='lower left')
plt.figure()
plt.plot(x[0],x[1],label='Scipy solver')
plt.plot(sol_X[:,0],sol_X[:,1],label='Implicit Euler')
plt.plot(a_sol_X[:,0], a_sol_X[:,1],label='Implicit Euler adaptive step')
plt.legend(loc='lower left')
plt.figure()