#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:37:39 2018

@author: carl
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from helperFunctions import *

#font_size = 15
#plt.rcParams['figure.figsize'] = (15,7)
#plt.rc('text',  usetex=True)
#plt.rc('font',   size=font_size)        # controls default text sizes
#plt.rc('axes',   titlesize=font_size)   # fontsize of the axes title
#plt.rc('axes',   labelsize=font_size)   # fontsize of the x any y labels
#plt.rc('xtick',  labelsize=font_size)   # fontsize of the tick labels
#plt.rc('ytick',  labelsize=font_size)   # fontsize of the tick labels
#plt.rc('legend', fontsize=font_size)    # legend fontsize
#plt.rc('figure', titlesize=font_size)   # size of the figure title


def VanDerPol (t,x,mu):
    # VANDERPOL Implementation of the Van der Pol model
    #
    # Syntax: xdot = VanDerPol(t,x,mu)
    xdot = np.zeros([2, 1])
    xdot[0] = x[1]
    xdot[1] = mu*(1-x[0]*x[0])*x[1]-x[0]
    return xdot

def JacVanDerPol(t,x,mu):
    # JACVANDERPOL Jacobian for the Van der Pol Equation
    #
    # Syntax: Jac = JacVanDerPol(t,x,mu)
    Jac = np.zeros([2, 2])
    Jac[1,0] = -2*mu*x[0]*x[1]-1.0
    Jac[0,1] = 1.0
    Jac[1,1] = mu*(1-x[0]*x[0])
    return Jac

def VanderPolfunjac(t,x,mu):
    return [VanDerPol(t,x,mu), JacVanDerPol(t,x,mu)]

mu = 10.0
x0 = np.asarray([2.0, 0.0])
t0 = 0
t1 = 5*mu

dt = 0.01


t,x = ImplicitEulerFixedStepSize(VanderPolfunjac,t0,t1,5,x0,mu)

r = ode(VanDerPol, JacVanDerPol).set_integrator('vode', method='bdf', order=15) 
r.set_initial_value(x0, t0).set_f_params(mu).set_jac_params(mu)

x = [[],[]]
t = np.arange(t0,t1+dt,dt)

while r.successful() and r.t < t1:
    xn = r.integrate(r.t+dt)
    x[0].append(xn[0])
    x[1].append(xn[1])


plt.plot(t, x[0])
plt.figure()
plt.plot(t, x[1])
plt.figure()
plt.plot(x[0],x[1])
#plt.show()


#options = odeset(’Jacobian’,@JacVanDerPol,’RelTol’,1.0e-6,’AbsTol’,1.0e-6)
#[T,X]=ode15s(@VanDerPol,[0 5*mu],x0,options,mu)