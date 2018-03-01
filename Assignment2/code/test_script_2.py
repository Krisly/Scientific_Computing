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


font_size = 15
plt.rcParams['figure.figsize'] = (15,7)
plt.rc('font',   size=font_size)       # controls default text sizes
plt.rc('axes',  titlesize=font_size)   # fontsize of the axes title
plt.rc('axes',   labelsize=font_size)  # fontsize of the x any y labels
plt.rc('xtick',  labelsize=font_size)  # fontsize of the tick labels
plt.rc('ytick',  labelsize=font_size)  # fontsize of the tick labels
plt.rc('legend', fontsize=font_size)   # legend fontsize
plt.rc('figure', titlesize=font_size)  # # size of the figure title

numsolv1 = numerical_solvers()
#vander_pol = numsolv1(1,3,5)
#print(sol1,sol2)
mu = 100
x0 = np.array([2.0,0.0])
t0 = 0
t1 = 10
dt = 0.01
t = np.arange(t0,t1+dt,dt)
absTol=10**(-8)
relTol=10**(-3)
epstol=0.8

sol_T,sol_X = numsolv1.ImplicitEulerFixedStepSize(numsolv1.VanderPolfunjac,
                                    t0,
                                    t1, 
                                    5000,
                                    x0.T,
                                    mu)

a_sol_T,a_sol_X,a_ss = numsolv1.ImplicitEulerAdaptiveStepSize(numsolv1.VanderPolfunjac,
                                                              t0,
                                                              t1,
                                                              x0.T,
                                                              dt,
                                                              absTol,
                                                              relTol,
                                                              epstol,
                                                              mu)

r = ode(numsolv1.VanDerPol,
        numsolv1.JacVanDerPol).set_integrator('vode',
                                              method='bdf',
                                              order=15)

r.set_initial_value(x0, t0).set_f_params(mu).set_jac_params(mu)

x = [[],[]]

while r.successful() and r.t < t1:
    xn = r.integrate(r.t+dt)
    x[0].append(xn[0])
    x[1].append(xn[1])


plt.plot(t, x[0],label='Scipy solver')
plt.plot(sol_T, sol_X[:,0],label='Implicit Euler fixed step')
plt.plot(a_sol_T, a_sol_X[:,0],label='Implicit Euler adaptive step')
plt.legend(loc='lower left')
plt.figure()
plt.plot(t, x[1],label='Scipy Solver')
plt.plot(sol_T, sol_X[:,1],label='Implicit Euler')
plt.plot(a_sol_T, a_sol_X[:,1],label='Implicit Euler adaptive step')
plt.legend(loc='lower left')
plt.figure()
plt.plot(x[0],x[1],label='Scipy solver')
plt.plot(sol_X[:,0],sol_X[:,1],label='Implicit Euler')
plt.plot(a_sol_X[:,0], a_sol_X[:,1],label='Implicit Euler adaptive step')
plt.legend(loc='lower left')
plt.figure()