#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 19:56:04 2018

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:38:58 2018

@author: kristoffer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

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
                         'coef4','coef5','coef6'])
    }

    N      = round((t[1]-t[0])/dt)
    n      = len(num_methods[method]['c'])
    k      = np.zeros((x.shape[0],n))
    absTol = 10**(-5)
    relTol = 10**(-5)
    epsTol = 0.8
    facmin = 0.1
    facmax = 5
    print(k)

    eee = ['Dormand-Prince']

    if (not (method in eee)) & (adap == False):
      print('Using fixed step size')

      X    = np.zeros((x.shape[0],N))
      X[:,0] = x
      T    = np.zeros((N))
      T[0] = t[0]

      for j in range(N-1):
       for i in range(n):
            k[:,i] = fun(T[j] + num_methods[method]['c'][i]*dt,
                         X[:,j] + dt*(np.sum(
                                  np.multiply(
                         np.asarray(num_methods[method]['coef{}'.format(i)]),
                                   k),axis=1)),kwargs)
    
       X[:,j+1] = X[:,j] + dt*np.sum(np.asarray(num_methods[method]['x'])*k,axis=1)
       T[j+1] = T[j] + dt
      return T,X

    elif method in eee:
      print('Using Embedded error estimator')
      T    = np.zeros((N))
      X    = np.zeros((x.shape[0],N))
      ss   = np.zeros((N))
      j    = 0
      px   = x
      pt   = t[0]
      while np.max(T) < t[1]:
        if(np.max(T)+dt>t[1]):
            dt = t[1]-np.max(T)
            
        AcceptStep = False
        while not AcceptStep:
          ts  = pt + dt
          k      = np.zeros((x.shape[0],n))
          for i in range(n):
            k[:,i] = fun(ts + num_methods[method]['c'][i]*dt,
                         px + dt*(np.sum(
                                  np.multiply(
                         np.asarray(num_methods[method]['coef{}'.format(i)]),
                                   k),axis=1)),kwargs)
    
          xs  = px + dt*np.sum(np.asarray(num_methods[method]['x'])*k,axis=1)
          #print(xs)
          xsh = px + dt*np.sum(np.asarray(num_methods[method]['xh'])*k,axis=1)
          #print(xsh)
          #print(ts,np.max(T))
          e   = np.abs(xs - xsh)
          num = absTol + np.abs(xs)*relTol
          r   = np.max(e/num)
          AcceptStep = (r <= 1)

          if AcceptStep:
            px       = xs
            #print(xs)
            pt       = ts
            #print(j+1,X.shape)
            X[:,j+1] = xs
            #print(X.shape)
            T[j+1]   = ts
            ss[j+1]  = dt
            j+=1
            if j+1==N:
              ap  = round(N/2)
              X  = np.append(X,np.zeros((xs.shape[0],ap)),axis=1)
              T  = np.append(T,np.zeros((ap)))
              ss = np.append(ss,np.zeros((ap)))
              N = N + ap

          dt = np.max([facmin,np.min([np.sqrt(epsTol/np.float64(r)),facmax])])*dt
          
      if j%10000==0:
          bs = X_C_A3.nbytes/1000000
          print("At time step: {}, with step size: {} \n Percentage of time executed: {} Size of sol array in mb: {}".format(ts,dt,(ts/t[1])*100,bs))
      
      return T[:j],X[:,:j],ss[:j]
    elif (not (method in eee)) & (adap == True):
      print('Using step doubling')
      
      T    = np.zeros((N))
      X    = np.zeros((x.shape[0],N))
      ss   = np.zeros((N))
      j    = 0
      px   = x
      pt   = t[0]
      k    = np.zeros((x.shape[0],n))
      k1   = np.zeros((x.shape[0],n))
      k2   = np.zeros((x.shape[0],n))
      
      while np.max(T) < t[1]:
        if(np.max(T)+dt>t[1]):
            dt = t[1]-np.max(T)
            
        AcceptStep = False
        while not AcceptStep:
            
          ts  = pt + dt

          for i in range(n):
            k[:,i] = fun(ts + num_methods[method]['c'][i]*dt,
                         px + dt*(np.sum(
                                  np.multiply(
                         np.asarray(num_methods[method]['coef{}'.format(i)]),
                                   k),axis=1)),kwargs)
          xs  = px + dt*np.sum(np.asarray(num_methods[method]['x'])*k,axis=1)

          tts  = pt + 0.5*dt

          for i in range(n):
            k1[:,i] = fun(tts + num_methods[method]['c'][i]*dt,
                         px + dt*(np.sum(
                                  np.multiply(
                         np.asarray(num_methods[method]['coef{}'.format(i)]),
                                   k1),axis=1)),kwargs)
          x_tmp = px + dt*np.sum(np.asarray(num_methods[method]['x'])*k1,axis=1)

          for i in range(n):
            k2[:,i] = fun(ts + num_methods[method]['c'][i]*dt,
                         x_tmp + dt*(np.sum(
                                  np.multiply(
                         np.asarray(num_methods[method]['coef{}'.format(i)]),
                                   k2),axis=1)),kwargs)
          x_tmp = x_tmp + dt*np.sum(np.asarray(num_methods[method]['x'])*k2,
                                    axis=1)
          
          e   = np.abs(xs - x_tmp)
          num = absTol + np.abs(x_tmp)*relTol
          r   = np.max(e/num)
          
          AcceptStep = (r <= 1)
          
          if AcceptStep:

            px       = x_tmp
            pt       = ts
            X[:,j+1] = xs
            T[j+1]   = ts
            ss[j+1]  = dt
            j+=1
            if j+1==N:
              ap  = round(N/2)
              X  = np.append(X,np.zeros((xs.shape[0],ap)),axis=1)
              T  = np.append(T,np.zeros((ap)))
              ss = np.append(ss,np.zeros((ap)))
              N = N + ap

          dt = np.max([facmin,np.min([np.sqrt(epsTol/np.float64(r)),
                       facmax])])*dt
    
          if j%10000==0:
              bs = X_C_A3.nbytes/1000000
              print("At time step: {}, with step size: {} \n Percentage of time executed: {} Size of sol array in mb: {}".format(ts,dt,(ts/t[1])*100,bs))
              #print('r: {} ts: {} dt: {} \n xs: {} x_tmp: {}'.format(r,
              #                                                   ts,
              #                                                   dt,
              #                                                   xs,
              #                                                   x_tmp))
          
      return T[:j],X[:,:j],ss[:j]
    else:
      print('Parameters not specified correctly')


def tf(t,x):
  return x

def true_tf(t):
  return -(2/(t**2-2))


T_C_3,X_C_3 = Runge_Kutta(VanDerPol,
                          np.array([0.5,0.5]),
                          [0,10],
                          0.001,
                          3,
                          method='Classic')

T_C_A3,X_C_A3,SS_C_A3 = Runge_Kutta(VanDerPol,
                          np.array([0.5,0.5]),
                          [0,10],
                          0.001,
                          3,
                          method='Classic',
                          adap=True)

T_DP_3,X_DP_3,SS_DP_3 = Runge_Kutta(VanDerPol,
                          np.array([0.5,0.5]),
                          [0,10],
                          0.001,
                          3,
                          method='Dormand-Prince')


plt.plot(T_C_3,X_C_3[1,:],label='RK4 FS')
plt.plot(T_C_A3,X_C_A3[1,:],label='RK4 AS')
plt.plot(T_DP_3,X_DP_3[1,:],label='DP54 AS')
plt.legend(loc='best')
plt.show()
plt.plot(T_C_3,X_C_3[0,:],label='RK4 FS')
plt.plot(T_C_A3,X_C_A3[0,:],label='RK4 AS')
plt.plot(T_DP_3,X_DP_3[0,:],label='DP54 AS')
plt.legend(loc='best')
plt.show()
plt.plot(X_C_3[0,:],X_C_3[1,:],label='RK4 FS')
plt.plot(X_C_A3[0,:],X_C_A3[1,:],label='RK4 AS')
plt.plot(X_DP_3[0,:],X_DP_3[1,:],label='DP54 AS')
plt.legend(loc='best')
plt.show()
plt.plot(T_C_A3,np.log(SS_C_A3),label='SS RK4')
plt.plot(T_DP_3,np.log(SS_DP_3),label='SS DP54')
plt.legend(loc='best')
plt.show()