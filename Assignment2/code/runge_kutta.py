#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:38:58 2018

@author: kristoffer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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





def Runge_Kutta(fun,x,t,dt,kwargs,method='Classic',adap=False):

    num_methods = {'Classic':
                pd.DataFrame(np.array([[0,1/2,1/2,1],
                                      [1/6,1/3,1/3,1/6],
                                      [0,1/2,0,0],
                                      [0,0,1/2,0],
                                      [0,0,0,1],
                                      [0,0,0,0]]).T,
                columns=['c', 'x','coef0', 'coef1', 'coef2', 'coef3']),
                '3/8-rule':
                pd.DataFrame(np.array([[0,1/3,2/3,1],
                                      [1/8,3/8,3/8,1/8],
                                      [0,1/3,-1/3,1],
                                      [0,0,1,-1],
                                      [0,0,0,1],
                                      [0,0,0,0]]).T,
                columns=['c', 'x','coef0', 'coef1', 'coef2', 'coef3']),
                'Dormand-Prince':
                pd.DataFrame(np.array([[0,1/5,3/10,4/5,8/9,1,1],
                 [35/384,0,500/1113,125/192,-2187/6784,11/84,0],
                 [5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40],
                 [0,1/5,3/40,44/45,19372/6561,9017/3168,35/384],
                 [0,0,9/40,-56/15,-25360/2187,-355/33,0],
                 [0,0,0,32/9,64448/6561,46732/5247,500/1113],
                 [0,0,0,0,-212/729,49/176,125/192],
                 [0,0,0,0,0,-5103/18656,-2187/6784],
                 [0,0,0,0,0,0,11/84],
                 [0,0,0,0,0,0,0]]).T,
                columns=['c', 'x','xh','coef0', 'coef1', 'coef2', 'coef3',
                         'coef4','coef5','coef6']),
                'Bogacki–Shampine':
                pd.DataFrame(np.array([[0,1/2,3/4,1],
                 [2/9,1/3,4/9,0],
                 [2/9,1/3,4/9,1/8],
                 [0,1/2,0,2/9],
                 [0,0,3/4,1/3],
                 [0,0,0,4/9],
                 [0,0,0,0]]).T,
                columns=['c', 'x','xh','coef0', 'coef1', 'coef2', 'coef3'])
    }

    N      = round((t[1]-t[0])/dt)
    n      = len(num_methods[method]['c'])
    k      = np.zeros((x.shape[0],n))
    absTol = 10**(-8)
    relTol = 10**(-8)
    epsTol = 0.8
    facmin = 0.1
    facmax = 5
    print(k)

    eee = ['Dormand-Prince']

    if (not (method in eee)) & (adap == False):
      print('Using fixed step size')

      X    = np.zeros((N))
      X[0] = x
      T    = np.zeros((N))
      T[0] = t[0]

      for j in range(N-1):
       for i in range(n):
          print(k[:,i])
          k[:,i] = fun(T[j] + num_methods[method]['c'][i]*dt,
                       X[:,j] + dt*(np.sum(num_methods[method]['coef{}'.format(i)]*k,axis=0)),kwargs)
    
       X[j+1] = X[j] + dt*np.sum(num_methods[method]['x']*k)
       T[j+1] = T[j] + dt
      return T,X

    elif method in eee:
      print('Using Embedded error estimator')
      T    = np.zeros((N))
      X    = np.zeros((N))
      ss   = np.zeros((N))
      j    = 0
      px   = x
      print(px)
      pt   = t[0]
      while np.max(T) <= t[1]:
        AcceptStep = False
        while not AcceptStep:
          for i in range(n):
            print(k[:,i])
            k[:,i] = fun(pt + num_methods[method]['c'][i]*dt,
                         px + dt*(np.sum(np.multiply(num_methods[method]['coef{}'.format(i)],k),axis=0)),kwargs)
    
          xs  = px + dt*np.sum(num_methods[method]['x']*k)
          xsh = px + dt*np.sum(num_methods[method]['xh']*k)
          ts  = pt + dt

          e = xs - xsh
          r = np.max(np.abs(e)/(absTol + xs*relTol))

          AcceptStep = (r <= 1)

          if AcceptStep:
            px      = xs
            pt      = ts
            X[j+1]  = xs
            T[j+1]  = ts
            ss[j+1] = dt
            j+=1
            if j+1==N:
              ap  = round(N/2)
              X  = np.append(X,np.zeros((ap)))
              T  = np.append(T,np.zeros((ap)))
              ss = np.append(ss,np.zeros((ap)))
              N = N + ap

          dt = np.max([facmin,np.min([np.sqrt(epsTol/np.float64(r)),facmax])])*dt 
      return T[:j],X[:j],ss[:j]
    elif (not (method in eee)) & (adap == True):
      print('Using step doubling')

    else:
      print('Parameters not specified correctly')


def tf(t,x):
  return x

def true_tf(t):
  return -(2/(t**2-2))

T,X,SS = Runge_Kutta(VanderPolfunjac,np.array([0.5,0.5]),[0,10],0.001,3,method='Dormand-Prince')

print(VanderPolfunjac(1,[0.5,0.5],3))
plt.plot(T,X,label='Runge-Kutta')
plt.plot(T,np.exp(T),label='True solution')
plt.legend(loc='best')
plt.show()