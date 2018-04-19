#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:34:45 2018

@author: carl
"""

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from Runge_Kutta_methods import Runge_Kutta
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

    sigma = 4
    x0 = [a, sigma]
    sol = []

    solver.set_initial_value(x0, t1).set_f_params(param)
    solver.integrate(t2)    
    sol = np.array(sol)
    
    resid = sol[-1,1] - b
    
    conv = False
    dsigma = None
    ntol = 1e-2
    it = 1
    while not conv:
        #print("asdasdasdasd")
        sigma_prev = sigma
        resid_prev = resid        
        #print(sigma)
        #print(resid)
        
        if (dsigma == None):
            sigma = 3
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
            
            #sigma = sigma + resid/(resid-resid_prev)
            #print(resid/(resid-resid_prev))
        sigma = sigma - resid/((resid-resid_prev)/(dsigma))
        dsigma = sigma - sigma_prev
        if (abs(resid)<ntol or it > 100):
            conv = True
        it += 1
    print ("iterations: " + str(it))
    return sigma
t1 = 0
t2 = 1
nsamp = 50
t = np.linspace(t1, t2, nsamp)
dt = (t2-t1)/nsamp

a = -1
b = 1.5
e = 0.1

xmean = 0.5*(t1+t2-a-b)
w0 = 0.5*(t1-t2+b-a)

x1_approx = t - xmean + w0*np.tanh(w0*(t-xmean)/(2*e))
x2_approx = 1 + (w0**2 * (1/np.cosh((w0 * (t - xmean))/(2 * e))**2))/(2 * e)

#plt.plot(t,x1_approx)
#plt.plot(t,x2_approx)
#plt.title('Provided approx')
#plt.show()

param = [a,b,e]

results = integrate.solve_bvp(lambda t,x: fun(t,x,param), lambda ua,ub: bc(ua,ub,param), t, [x1_approx, x2_approx])

#plt.plot(results.x, results.y[0])
#plt.plot(results.x, results.y[1])
#plt.title('scipy solve_bvp')
#plt.show()


sigma = single_shoot(fun, [a, b], [t1, t2], param)

x0 = np.array([a, sigma])

solver = ode(fun).set_integrator('dopri5')

sol = []
def solout(t, y):
    sol.append([t, *y])
solver.set_solout(solout)
solver.set_initial_value(x0, t1).set_f_params(param)
solver.integrate(t2)

sol = np.array(sol)
#plt.plot(sol[:,0], sol[:,1])
#plt.plot(sol[:,0], sol[:,2])
#plt.title('DOPRI IVP, x=' + str(x0[0]) + ", x'=" + str(x0[1]))
#plt.show()



def Derivatives(t,x,p):
    e = p[2]
    dfdx = np.array([0, 1, -(x[1]-1)/e, -x[0]/e])
    dfdp = np.array([0, 0])
    return dfdx, dfdp

def modelAndSens(t,z,p):
    x = z[:-2]
    sp = np.array(z[-2:])
    #sp = np.reshape(sp,[2,2])
    xdot = fun(t,x,p)
    dfdx, dfdp = Derivatives(t,x,p)
    dfdx = np.reshape(dfdx,[2,2])
    #print(np.asmatrix(dfdx) * np.asmatrix(sp).T)
    Spdot = (np.asmatrix(dfdx)*np.asmatrix(sp).T).T + dfdp
    zdot = np.concatenate([xdot, np.asarray(Spdot.flatten()).flatten()])
    #print(zdot)
    #print(type(zdot))
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

plt.figure(figsize=(13,5))
plt.plot(t,x1_approx)
plt.plot(results.x, results.y[0],"g--")
plt.plot(sol[:,0], sol[:,1])
plt.title("x1 (u)")
plt.legend(["Approximated","scipy solve_bvp","shooting method"])
plt.xlabel("t")
plt.ylabel("x1")
plt.show()

plt.figure(figsize=(13,5))
plt.plot(t,x2_approx)
plt.plot(results.x, results.y[1],"g--")
plt.plot(sol[:,0], sol[:,2])
plt.title("x2 (u')")
plt.legend(["Approximated","scipy solve_bvp","shooting method"])
plt.xlabel("t")
plt.ylabel("x2")
plt.show()

plt.figure(figsize=(13,5))
plt.plot(sol[:,0], sol[:,3])
plt.plot(sol[:,0], sol[:,4])
plt.title("Sensitivities")
plt.legend(["dx1/ds","dx2/ds"])
plt.xlabel("t")
plt.ylabel("S")
plt.show()

print(sol[-1,3])