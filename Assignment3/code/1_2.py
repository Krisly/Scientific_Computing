#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:34:45 2018

@author: carl
"""

import numpy as np
from scipy import integrate
import matplotlib
from matplotlib import pyplot as plt
from scipy.integrate import ode

# lec 14 - 24

def fun(t,x,p):
    e = p[2]
    dxdt = np.array([x[1], -x[0]*(x[1]-1)/e])
    return dxdt

def bc(ua, ub, p):
    return np.array([ua[0]-p[0], ub[0]-p[1]])

def single_shoot(fun, bcond, t, param):
    t1 = t[0]
    t2 = t[1]
    a = bcond[0]
    b = bcond[1]

    solver = ode(fun).set_integrator('dopri5')
    def solout(t, y):
        sol.append([t, *y])
    solver.set_solout(solout)

    sigma = a
    x0 = [a, sigma]
    sol = []

    solver.set_initial_value(x0, t1).set_f_params(param)
    solver.integrate(t2)    
    sol = np.array(sol)
    
    resid = sol[-1,1] - b
    
    conv = False
    dsigma = None
    ntol = 1e-4
    it = 1
    while not conv:
        sigma_prev = sigma   
        ysprev = sol[-1,1]
        if (it == 1):
            sigma = 2*a
            dsigma = sigma - sigma_prev
            x0 = [a, sigma]
            sol = []
            
            solver.set_initial_value(x0, t1).set_f_params(param)
            solver.integrate(t2)    
            sol = np.array(sol)
            
            resid = sol[-1,1] - b
        else:
            x0 = [a, sigma]
            sol = []
            
            solver.set_initial_value(x0, t1).set_f_params(param)
            solver.integrate(t2)    
            sol = np.array(sol)
            
            resid = sol[-1,1] - b
        dys = sol[-1,1] - ysprev
        print(it, sigma, resid, dys, dsigma)
        sigma = sigma - resid/((dys)/(dsigma))/2
        dsigma = sigma - sigma_prev
        if (abs(resid)<ntol or it > 100):
            conv = True
        it += 1
    print ("iterations: " + str(it))
    return sigma
t1 = 0
t2 = 1
nsamp = 1000
t = np.linspace(t1, t2, nsamp)
dt = (t2-t1)/nsamp

a = -1
b = 1.5
e = 0.1

xmean = 0.5*(t1+t2-a-b)
w0 = 0.5*(t1-t2+b-a)

x1_approx = t - xmean + w0*np.tanh(w0*(t-xmean)/(2*e))
x2_approx = 1 + (w0**2 * (1/np.cosh((w0 * (t - xmean))/(2 * e))**2))/(2 * e)

param = [a,b,e]

results = integrate.solve_bvp(lambda t,x: fun(t,x,param), lambda ua,ub: bc(ua,ub,param), t, [x1_approx, x2_approx])

sigma = single_shoot(fun, [a, b], [t1, t2], param)
print(sigma)
x0 = np.array([a, sigma])

solver = ode(fun).set_integrator('dopri5')

sol = []
def solout(t, y):
    sol.append([t, *y])
solver.set_solout(solout)
solver.set_initial_value(x0, t1).set_f_params(param)
solver.integrate(t2)

sol = np.array(sol)


def Derivatives(t,x,p):
    e = p[2]
    dfdx = np.array([0, 1, -(x[1]-1)/e, -x[0]/e])
    dfdp = np.array([0, 0])
    return dfdx, dfdp

def modelAndSens(t,z,p):
    x = z[:-2]
    sp = np.array(z[-2:])
    xdot = fun(t,x,p)
    dfdx, dfdp = Derivatives(t,x,p)
    dfdx = np.reshape(dfdx,[2,2])
    Spdot = (np.asmatrix(dfdx)*np.asmatrix(sp).T).T + dfdp
    zdot = np.concatenate([xdot, np.asarray(Spdot.flatten()).flatten()])
    return zdot

def qwe(t,x,p):
    return 5

x0 = np.array([a, sigma, 0, 1])


solver = ode(modelAndSens).set_integrator('dopri5')

sol = []
def solout(t, y):
    sol.append([t, *y])
solver.set_solout(solout)
print (x0)
print (t1)
print (param)
solver.set_initial_value(x0, t1).set_f_params(param)
solver.integrate(t2)

sol = np.array(sol)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

plt.figure(figsize=(15,5))
plt.plot(t,x1_approx, linewidth=2.0)
plt.plot(sol[:,0], sol[:,1], linewidth=2.0)
plt.plot(results.x, results.y[0],"g--", linewidth=2.0)
plt.title("$x_1 (u)$")
plt.legend(["Approximated","shooting method","scipy solve_bvp"])
plt.xlabel("t")
plt.ylabel("$x_1$")
plt.savefig('1-2-1.eps', format='eps', dpi=500, bbox_inches='tight')
plt.show()

plt.figure(figsize=(15,5))
plt.plot(t,x2_approx, linewidth=2.0)
plt.plot(sol[:,0], sol[:,2], linewidth=2.0)
plt.plot(results.x, results.y[1],"g--", linewidth=2.0)
plt.title("$x_2 (u')$")
plt.legend(["Approximated","shooting method","scipy solve_bvp"])
plt.xlabel("t")
plt.ylabel("$x_2$")
plt.savefig('1-2-2.eps', format='eps', dpi=500, bbox_inches='tight')
plt.show()

plt.figure(figsize=(15,5))
plt.plot(sol[:,0], sol[:,3], linewidth=2.0)
plt.plot(sol[:,0], sol[:,4], linewidth=2.0)
plt.title("Sensitivities")
plt.legend(["$\\frac{\\partial x_1}{\\partial \\sigma}$","$\\frac{\\partial x_2}{\\partial \\sigma}$"],fontsize=30)
plt.xlabel("t")
plt.ylabel("S")
plt.savefig('1-2-3.eps', format='eps', dpi=500, bbox_inches='tight')
plt.show()

print(sol[-1,3])