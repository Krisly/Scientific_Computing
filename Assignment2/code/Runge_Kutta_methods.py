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
from scipy.integrate import ode
import cProfile

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

def rk_step(fun,num_methods,method,k,t,x,dt,xm,kwargs):
	for i in range(len(num_methods[method]['c'])):
		k[:,i] = fun(t + num_methods[method]['c'][i]*dt,
        			 x + dt*(np.sum(np.asarray(num_methods[method]['coef{}'.format(i)])*k,axis=1)),
                     kwargs)
	return k, x + dt*np.sum(np.asarray(num_methods[method][xm])*k,axis=1)


def fill_array(C,A,b,a):
		nrow = A.shape[0]
		ncol = A.shape[1]
	
		for i in range(len(a)):
			for k in range(len(a)):
				z      = a[i,k]
				I      = np.eye(nrow,ncol)
				e 	   = np.ones((nrow,1))
				eiAinv = np.linalg.inv(I-z*A)
				c_tmp  = 1 + z*b.T*eiAinv*e
				if np.absolute(c_tmp) > 1:
					C[k,i] = 1
				else:
					C[k,i] = np.absolute(c_tmp)
		return C

def plot_stab_reg(C,A,b):

	xmin = -5
	xmax = 5
	ymin = -5
	ymax = 5
	xv, yv = np.meshgrid(np.arange(xmin,xmax,0.05), np.arange(ymin,ymax,0.05), sparse=False, indexing='ij')
	
	a = xv + 1j*yv
	
	t = fill_array(C,A,b,a)
	#print(t/np.amax(t))
	#plt.scatter(a.real,a.imag, c = t,cmap='hsv')
	plt.imshow(t,cmap='hsv',extent=[xmin, xmax, ymax, ymin],interpolation='bilinear')
	plt.colorbar()
	plt.show()

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
                 [0,0,0,0,0,0,0]],dtype=np.float64).T,
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
    absTol = 10**(-4)
    relTol = 10**(-4)
    epsTol = 0.8
    facmin = 0.1
    facmax = 5

    eee      = ['Dormand-Prince', 'Bogacki–Shampine','ESDIRK23']
    implicit = ['ESDIRK23']

    if (not (method in eee)) & (adap == False):
      print('Using fixed step size for: {}'.format(method))

      X    = np.zeros((x.shape[0],N))
      X[:,0] = x
      T    = np.zeros((N))
      T[0] = t[0]

      for j in range(N-1):
       k,xs   = rk_step(fun,num_methods,method,k,T[j],X[:,j],dt,'x',kwargs)
       X[:,j+1] = xs
       T[j+1] = T[j] + dt

       if j%round(N/15)==0:
            bs = X.nbytes/1000000
            print("{:<8} {:<8} {:<8} {:<8} \n {:>8} {:>17} {:>11} {:>15}".format('Time step:',
            																  'Step size[power]:',
            																  'Percentage:',
            																  'Array size[mb]:',
            															      round(T[j],2),
              																  round(np.log10(dt),2),
              																  round((T[j]/t[1])*100,2),
              																  bs))
      return T,X

    elif method in eee:
      print('Using Embedded error estimator for: {}'.format(method))
      T      = np.zeros((N))
      X      = np.zeros((x.shape[0],N))
      k      = np.zeros((x.shape[0],n))
      ss     = np.zeros((N))
      j      = 0
      X[:,0] = x

      while T[j] < t[1]:
        if(T[j]+dt>t[1]):
            dt = t[1]-T[j]
            
        AcceptStep = False
        while not AcceptStep:
          if method == 'Dormand-Prince':
            k,xs   = rk_step(fun,num_methods,method,k,T[j],X[:,j],dt,'xh',kwargs)
            na,xsh = rk_step(fun,num_methods,method,k,T[j],X[:,j],dt,'x',kwargs)
          else:
            k,xs   = rk_step(fun,num_methods,method,k,T[j],X[:,j],dt,'x',kwargs)
            na,xsh = rk_step(fun,num_methods,method,k,T[j],X[:,j],dt,'xh',kwargs)

          e   = np.abs(xs - xsh)
          num = absTol + np.abs(xs)*relTol
          r   = np.max(e/num)
          AcceptStep = (r <= 1)

          if AcceptStep:
            X[:,j+1] = xs
            T[j+1]   = T[j] + dt 
            ss[j+1]  = dt

            j+=1
            if j+1==N:
              ap  = round(N/2)
              X  = np.append(X,np.zeros((xs.shape[0],ap)),axis=1)
              T  = np.append(T,np.zeros((ap)))
              ss = np.append(ss,np.zeros((ap)))
              N = N + ap

          if j%(len(X)/15)==0:
            bs = X.nbytes/1000000
            print("{:<8} {:<8} {:<8} {:<8} \n {:>8} {:>17} {:>11} {:>15}".format('Time step:',
            																  'Step size[power]:',
            																  'Percentage:',
            																  'Array size[mb]:',
            															      round(T[j],2),
              																  round(np.log10(dt),2),
              																  round((T[j]/t[1])*100,2),
              																  bs))

          dt = np.max([facmin,np.min([np.sqrt(epsTol/np.float64(r)),facmax])])*dt
          

      return T[:j],X[:,:j],ss[:j]
    elif (not (method in eee)) & (adap == True):
      print('Using step doubling for: {}'.format(method))
      
      T      = np.zeros((N))
      X      = np.zeros((x.shape[0],N))
      ss     = np.zeros((N))
      j      = 0
      X[:,0] = x
      T[0]   = t[0]
      k      = np.zeros((x.shape[0],n))

      while T[j] < t[1]:
        if(T[j]+dt>t[1]):
            dt = t[1]-T[j]
            
        AcceptStep = False
        while not AcceptStep:
            
          k,xs    = rk_step(fun,num_methods,method,k,T[j],X[:,j],dt,'x',kwargs)
          k,x_tmp = rk_step(fun,num_methods,method,k,T[j],X[:,j],0.5*dt,'x',kwargs)
          k,x_tmp = rk_step(fun,num_methods,method,k,T[j],x_tmp,0.5*dt,'x',kwargs)
          
          e   = np.abs(xs - x_tmp)
          num = absTol + np.abs(x_tmp)*relTol
          r   = np.max(e/num)
          
          AcceptStep = (r <= 1)
          
          if AcceptStep:
            X[:,j+1] = x_tmp
            T[j+1]   = T[j] + dt
            ss[j+1]  = dt
            j+=1
            if j+1==N:
              ap = round(N/2)
              X  = np.append(X,np.zeros((xs.shape[0],ap)),axis=1)
              T  = np.append(T,np.zeros((ap)))
              ss = np.append(ss,np.zeros((ap)))
              N  = N + ap

          dt = np.max([facmin,np.min([np.sqrt(epsTol/np.float64(r)),
                       facmax])])*dt
    
          if j%(len(X)/15)==0:
              bs = X.nbytes/1000000
              print("{:<8} {:<8} {:<8} {:<8} \n {:>8} {:>17} {:>11} {:>15}".format('Time step:',
            																  'Step size[power]:',
            																  'Percentage:',
            																  'Array size[mb]:',
            															      round(T[j],2),
              																  round(np.log10(dt),2),
              																  round((T[j]/t[1])*100,2),
              																  bs))
      return T[:j],X[:,:j],ss[:j]
    elif(method in implicit) & (method in eee):
    	print('Nice')

    else:
      print('Parameters not specified correctly')


def tf(t,x):
  return x

def true_tf(t):
  return np.exp(t)

abstol = 10**(-8)
reltol = 10**(-8)
x0 = np.array([0.5,0.5])
dt = 10**(-2)
mu = 3
ti  = [0,100]
T_C_3,X_C_3 = Runge_Kutta(VanDerPol,
                          x0,
                          ti,
                          dt,
                          mu,
                          method='Classic')

T_C_A3,X_C_A3,SS_C_A3 = Runge_Kutta(VanDerPol,
                          x0,
                          ti,
                          dt,
                          mu,
                          method='Classic',
                          adap=True)

T_DP_3,X_DP_3,SS_DP_3 = Runge_Kutta(VanDerPol,
                          x0,
                          ti,
                          dt,
                          mu,
                          method='Dormand-Prince')

T_BS_3,X_BS_3,SS_BS_3 = Runge_Kutta(VanDerPol,
                          x0,
                          ti,
                          dt,
                          mu,
                          method='Bogacki–Shampine')


r = ode(VanDerPol,
        JacVanDerPol).set_integrator('vode',
                                      method='bdf',
                                      with_jacobian=True,
                                      order=15)
r.set_initial_value(x0, ti[0]).set_f_params(mu).set_jac_params(mu)
x_sci_s = [[],[]]
t       = np.arange(0,ti[1]+0.01,0.01)
# Solving using the scipy solver
while r.successful() and r.t < ti[1]:
    xn = r.integrate(r.t+0.01)
    x_sci_s[0].append(xn[0])
    x_sci_s[1].append(xn[1])

fig, ax = plt.subplots(3, 1, figsize=(15,10), sharex=False)
# Plotting the results
ax[0].plot(t[:len(x_sci_s[0][:])],x_sci_s[0][:],label='Scipy')
ax[0].plot(T_C_3,X_C_3[0,:],label='RK4 FS')
ax[0].plot(T_C_A3,X_C_A3[0,:],label='RK4 AS')
ax[0].plot(T_DP_3,X_DP_3[0,:],label='DP54 AS')
ax[0].plot(T_BS_3,X_BS_3[0,:],label='BS AS')
ax[0].set_title(r'Phase state of the Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(dt,mu,abstol,reltol))
ax[0].set_xticks([])
ax[0].set_ylim(-5,5)
ax[0].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)

ax[1].plot(t[:len(x_sci_s[1][:])],x_sci_s[1][:],label='Scipy')
ax[1].plot(T_C_3,X_C_3[1,:],label='RK4 FS')
ax[1].plot(T_C_A3,X_C_A3[1,:],label='RK4 AS')
ax[1].plot(T_DP_3,X_DP_3[1,:],label='DP54 AS')
ax[1].plot(T_BS_3,X_BS_3[1,:],label='BS AS')
ax[1].set_title(r'Phase state of the Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(dt,mu,abstol,reltol))
ax[1].set_xticks([])
ax[1].set_ylim(-5,5)
ax[1].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)

ax[2].plot(T_C_A3[1:],np.log(SS_C_A3[1:]),label='SS RK4')
ax[2].plot(T_DP_3[1:],np.log(SS_DP_3[1:]),label='SS DP54')
ax[2].plot(T_BS_3[1:],np.log(SS_BS_3[1:]),label='SS BS')
ax[2].set_title('Semi log-plot of step sizes with tolerance {}'.format(10**(-4)))
ax[2].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)
plt.show()

plt.figure()
plt.plot(x_sci_s[0][:],x_sci_s[1][:],label='Scipy')
plt.plot(X_C_3[0,:],X_C_3[1,:],label='RK4 FS')
plt.plot(X_C_A3[0,:],X_C_A3[1,:],label='RK4 AS')
plt.plot(X_DP_3[0,:],X_DP_3[1,:],label='DP54 AS')
plt.plot(X_BS_3[0,:],X_BS_3[1,:],label='BS AS')
plt.title(r'Phase state of the Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(dt,mu,abstol,reltol))
plt.legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)
plt.show()




cProfile.run("Runge_Kutta(VanDerPol,np.array([0.5,0.5]),[0,10],10**(-4),mu,method='Bogacki–Shampine',adap=True)",sort='cumtime')



