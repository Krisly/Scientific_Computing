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
mu              = 10
x0              = np.array([0.5,0.5])
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

def PreyPredatorfunjac(self, t,x,params):
    return PreyPredator(t,x,params), JacPreyPredator(t,x,params)

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

fig, ax = plt.subplots(3, 1, figsize=(15,10), sharex=False)
# Plotting the results
ax[0].plot(t[:len(x[0])], x[0],label='Scipy')
ax[0].plot(sol_T, sol_X[:,0],label='IE FS')
ax[0].plot(a_sol_T, a_sol_X[:,0],label='IE AS')
ax[0].set_xticks([])
ax[0].set_title(r'Plot of state one Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(numsolv1.dt,mu,absTol_as,numsolv1.relTol))
ax[0].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)

ax[1].plot(t[:len(x[0])], x[1],label='Scipy')
ax[1].plot(sol_T, sol_X[:,1],label='IE FS')
ax[1].plot(a_sol_T, a_sol_X[:,1],label='IE AS')
ax[1].set_title(r'Plot of state two Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(numsolv1.dt,mu,absTol_as,numsolv1.relTol))
ax[1].set_xticks([])
ax[1].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)

ax[2].plot(a_sol_T,np.log(a_ss),label='SS')
ax[2].set_title(r'Semi log-plot of step sizes with tolerance {}'.format(absTol_as))
ax[2].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)
plt.savefig('./figs/VanDerPol_IE_mu{}_abstol{}_reltol{}.pdf'.format(round(mu),
																  (-1)*round(np.log10(absTol_as)),
																  (-1)*round(np.log10(numsolv1.relTol))))
plt.show()

plt.figure()
plt.plot(x[0], x[1],label='Scipy')
plt.plot(a_sol_X[:,0], a_sol_X[:,1],label='IE FS')
plt.plot(sol_X[:,0], sol_X[:,1],label='IE AS')
plt.title(r'Phase state plot. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(numsolv1.dt,mu,absTol_as,numsolv1.relTol))
plt.legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)
plt.savefig('./figs/VanDerPol_phase_IE_mu{}_abstol{}_reltol{}.pdf'.format(round(mu),
																  (-1)*round(np.log10(absTol_as)),
																  (-1)*round(np.log10(numsolv1.relTol))))
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
'''