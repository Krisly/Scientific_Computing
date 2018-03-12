#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:37:39 2018

@author: carl
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import helperFunctions as hF

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


def VanDerPol(t,x,mu):
    # VANDERPOL Implementation of the Van der Pol model
    #
    # Syntax: xdot = VanDerPol(t,x,mu)
    xdot = np.zeros([2, 1])
    xdot[0] = x[1]
    xdot[1] = mu*(1-x[0]*x[0])*x[1]-x[0]
    return xdot.T[0]

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


def PreyPredator(t,x,params):
    # PreyPredator Implementation of the PreyPredator model
    #
    # Syntax: xdot = VanDerPol(t,x,params)
    a = params[0]
    b = params[1]
    xdot = np.zeros([2, 1])
    xdot[0] = a*(1-x[1])*x[0]
    xdot[1] = -b*(1-x[0])*x[1]
    return xdot.T[0]

def JacPreyPredator(t,x,params):
    # JACPREYPREDATOR Jacobian for the Prey Predator Equation
    #
    # Syntax: Jac = JacPreyPredator(t,x,params)
    a = params[0]
    b = params[1]
    Jac = np.zeros([2, 2])
    Jac[0,0] = a*(1-x[1])
    Jac[1,0] = b*x[1]
    Jac[0,1] = -a*x[0]
    Jac[1,1] = -b*(1-x[0])
    return Jac

def PreyPredatorfunjac(t,x,params):
    return [PreyPredator(t,x,params), JacPreyPredator(t,x,params)]


mu = 10
x0 = np.array([[0.5], [0.5]])
t0 = 0
t1 = 5*mu

dt = 0.001

r = ode(VanDerPol, JacVanDerPol).set_integrator('vode', method='bdf', order=15) 
r.set_initial_value(x0, t0).set_f_params(mu).set_jac_params(mu)

x_scipy = [[],[]]
t_scipy = np.arange(t0,t1+dt,dt)

while r.successful() and r.t < t1:
    xns = r.integrate(r.t+dt)
    x_scipy[0].append(xns[0].item())
    x_scipy[1].append(xns[1].item())


#plt.plot(t, x[0])
#plt.figure()
#plt.plot(t, x[1])
#plt.figure()
#plt.plot(x[0],x[1])
#plt.show()

[t, x] = hF.ExplicitEulerFixedStepSize(VanDerPol, t0, t1, round((t1-t0)/dt),x0,mu)

'''
plt.plot(t, x[:,0])
plt.figure()
plt.plot(t, x[:,1])
plt.figure()
plt.plot(x[:,0],x[:,1])
plt.show()
'''
abstol = 1e-1
reltol = 1e-1

[t2, x2] = hF.ExplicitEulerAdaptiveStep(VanDerPol,[t0, t1],[2,0],dt,abstol,reltol,mu)

'''
plt.plot(t2, x2[:,0], '-o')
plt.figure()
plt.plot(t2, x2[:,1], '-o')
plt.figure()
plt.plot(x2[:,0],x2[:,1], '-o')
plt.show()
'''


fig, ax = plt.subplots(3, 1, figsize=(15,10), sharex=False)
# Plotting the results
ax[0].plot(t_scipy, x_scipy[0][:],label='Scipy')
ax[0].plot(t, x[:,0],label='EE FS')
ax[0].plot(t2, x2[:,0],'-o',label='EE AS')
ax[0].set_title(r'Plot of state one Van Der Pol. [SS: {} $\mu = {}$ abstol = {} reltol = {}]'.format(round((t1-t0)/dt),mu,abstol,reltol))
ax[0].legend(bbox_to_anchor=(-0.12, 1), loc=2, borderaxespad=0.)

ax[1].plot(t_scipy, x_scipy[1][:],label='Scipy')
ax[1].plot(t, x[:,1],label='EE FS')
ax[1].plot(t2, x2[:,1],'-o',label='EE AS')
ax[1].set_title(r'Plot of state two Van der Pol. [SS: {} $\mu = {}$ abstol = {} reltol = {}]'.format(round((t1-t0)/dt),mu,abstol,reltol))
ax[1].legend(bbox_to_anchor=(-0.12, 1), loc=2, borderaxespad=0.)

ax[2].plot(x_scipy[0][:], x_scipy[1][:],label='Scipy')
ax[2].plot(x[:,0], x[:,1],label='EE FS')
ax[2].plot(x2[:,0], x2[:,1],'-o',label='EE AS')
ax[2].set_title(r'Phase state plot. [SS: {} $\mu = {}$ abstol = {} reltol = {}]'.format(round((t1-t0)/dt),mu,abstol,reltol))
ax[2].legend(bbox_to_anchor=(-0.12, 1), loc=2, borderaxespad=0.)
plt.savefig('./figs/VanDerPol_EE_mu{}.pdf'.format(round(mu)))
plt.show()
#options = odeset(’Jacobian’,@JacVanDerPol,’RelTol’,1.0e-6,’AbsTol’,1.0e-6)
#[T,X]=ode15s(@VanDerPol,[0 5*mu],x0,options,mu)

a = 1
b = 1
x0 = [2,2]


r = ode(PreyPredator, JacPreyPredator).set_integrator('vode', method='bdf', order=15) 
r.set_initial_value(x0, t0).set_f_params([a,b]).set_jac_params([a,b])

x_scipy = [[],[]]
t_scipy = np.arange(t0,t1+dt,dt)

while r.successful() and r.t < t1:
    xns = r.integrate(r.t+dt)
    x_scipy[0].append(xns[0].item())
    x_scipy[1].append(xns[1].item())


[t, x] = hF.ExplicitEulerFixedStepSize(PreyPredator, t0, t1, round((t1-t0)/dt),x0,[a,b])
'''
plt.plot(t, x[:,0])
plt.figure()
plt.plot(t, x[:,1])
plt.figure()
plt.plot(x[:,0],x[:,1])
plt.show()
'''
abstol = 1e-3
reltol = 1e-3

[t2, x2] = hF.ExplicitEulerAdaptiveStep(PreyPredator,[t0, t1],x0,dt,abstol,reltol,[a,b])
'''
plt.plot(t2, x2[:,0], '-o')
plt.figure()
plt.plot(t2, x2[:,1], '-o')
plt.figure()
plt.plot(x2[:,0],x2[:,1], '-o')
plt.show()
'''

fig, ax = plt.subplots(3,1, figsize=(15,10), sharex=False)
# Plotting the results
ax[0].plot(t_scipy, x_scipy[0][:],label='Scipy')
ax[0].plot(t, x[:,0],label='EE FS')
ax[0].plot(t2, x2[:,0],'-o',label='EE AS')
ax[0].set_title(r'Plot of state one Predator Prey. [SS: {} $\alpha = {}$, $\beta = {}$, abstol = {} reltol = {}]'.format(round((t1-t0)/dt),a,b,abstol,reltol))
ax[0].legend(bbox_to_anchor=(-0.1, 1), loc=2, borderaxespad=0.)

ax[1].plot(t_scipy, x_scipy[1][:],label='Scipy')
ax[1].plot(t, x[:,1],label='EE FS')
ax[1].plot(t2, x2[:,1],'-o',label='EE AS')
ax[1].set_title(r'Plot of state two Predator Prey. [SS: {} $\alpha = {}$, $\beta = {}$, abstol = {} reltol = {}]'.format(round((t1-t0)/dt),a,b,abstol,reltol))
ax[1].legend(bbox_to_anchor=(-0.1, 1), loc=2, borderaxespad=0.)

ax[2].plot(x_scipy[0][:], x_scipy[1][:],label='Scipy')
ax[2].plot(x[:,0], x[:,1],label='EE FS')
ax[2].plot(x2[:,0], x2[:,1],'-o',label='EE AS')
ax[2].set_title(r'Phase state plot. [SS: {} $\alpha = {}$, $\beta = {}$, abstol = {} reltol = {}]'.format(round((t1-t0)/dt),a,b,abstol,reltol))
ax[2].legend(bbox_to_anchor=(-0.1, 1), loc=2, borderaxespad=0.)
plt.savefig('./figs/PredatorPrey_EE_a{}_b{}.pdf'.format(round(a),round(b)))
plt.show()


