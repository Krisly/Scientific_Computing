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
import scipy

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

def testEqn(t,x,p):
    lamda = p[0]
    return lamda*x

def JacTestEqn(t,x,p):
    return p[0]

def testEqnFunJac(t,x,p):
    return testEqn(t,x,p), JacTestEqn(t,x,p)

def testEqnAnalyt(t,x0,p):
    lamda = p[0]
    return x0*np.exp(lamda*t)

def rk_step(fun,num_methods,method,k,t,x,dt,xm,kwargs):
  # general RK steping function 
  for i in range(len(num_methods[method]['c'])):
    k[:,i] = fun(t + num_methods[method]['c'][i]*dt,
                  x + dt*(np.sum(np.asarray(num_methods[method]['coef{}'.format(i)])*k,axis=1)),kwargs)
  return k, x + dt*np.sum(np.asarray(num_methods[method][xm])*k,axis=1)


def rk_step_impl(fun,jac,num_methods,method,tn,xn,dt,kwargs):
    # general implicit RK stepping function
    c = num_methods[method]['c']
    b = np.asarray(num_methods[method]['x'])
    a = np.zeros([len(c), len(b)])

    tol = 1e-5
    maxit = 50
    
    X = np.zeros([np.size(xn), len(c)])
    Xn = np.zeros([np.size(xn), len(c)])
    F = np.zeros([np.size(xn), len(c)])
    T = np.zeros(len(c))
    T = tn + c*dt
    J = jac(T,xn,kwargs)
    
    # variable initilizations
    for i in range(len(c)):
        X[:,i] = xn
        Xn[:,i] = xn
        F[:,i] = fun(T[i], X[:,i], kwargs)
        a[:,i] = np.asarray(num_methods[method]['coef{}'.format(i)])

    R = np.zeros([len(xn), len(c)])
    Fs = np.array([F[:,q] for q in range(len(c))])

    # intial residual calculation
    for i in range(len(c)):
        R[:,i] = X[:,i] - Xn[:,i] - dt*np.sum(a[:,i]*Fs.T,axis = 1).T
    
    huge = np.concatenate([np.concatenate([a[q,i]*J for q in range(len(c))],axis=1) for i in range(len(c))])
    dRdx = np.eye(len(c)*len(xn)) - dt*huge

    it = 1
    R = (np.concatenate(R.T))

    # Newton's method main loop
    while ( (np.linalg.norm(R,np.inf) > tol) and (it <= maxit) ):
        mdx = np.linalg.solve(dRdx,R.T)
        mdx = (np.reshape(mdx, [len(c), len(xn)]).T)
        X = X - mdx
        for i in range(len(c)):
            F[:,i] = fun(T[i], X[:,i], kwargs)
        Fs = np.array([F[:,q] for q in range(len(c))])
        R = np.zeros([len(xn), len(c)])
        for i in range(len(c)):
            R[:,i] = X[:,i] - Xn[:,i] - dt*np.sum(a[:,i]*Fs.T,axis = 1).T
        R = (np.concatenate(R.T))
        huge = np.concatenate([np.concatenate([a[q,i]*J for q in range(len(c))],axis=1) for i in range(len(c))])
        dRdx = np.eye(len(c)*len(xn)) - dt*huge
        it += 1
    return T[len(c)-1], X[:,len(c)-1]

def fill_array(C,A,b,a):
          # helper function for the stability region plotting
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
     # stability region plotting
	xmin = -5
	xmax = 5
	ymin = -5
	ymax = 5
	xv, yv = np.meshgrid(np.arange(xmin,xmax,0.05), np.arange(ymin,ymax,0.05), sparse=False, indexing='ij')
	
	a = xv + 1j*yv
	
	t = fill_array(C,A,b,a)
	plt.imshow(t,cmap='hsv',extent=[xmin, xmax, ymax, ymin],interpolation='bilinear')
	plt.colorbar()
	plt.show()


def Runge_Kutta(fun,x,t,dt,kwargs,method='Classic',adap=False,jac=None):
    # Runge-Kutta Swiss army knife
    
    # declaring the various methods and coefficients
    num_methods = {'Classic':
                pd.DataFrame(np.array([[0,1/2,1/2,1],
                                      [1/6,1/3,1/3,1/6],
                                      [0,0,0,0],
                                      [1/2,0,0,0],
                                      [0,1/2,0,0],
                                      [0,0,1,0]]).T,
                columns=['c', 'x','coef0', 'coef1', 'coef2', 'coef3']),
                '3/8-rule':
                pd.DataFrame(np.array([[0,1/3,2/3,1],
                                      [1/8,3/8,3/8,1/8],
                                      [0,0,0,0],
                                      [1/3,0,0,0],
                                      [-1/3,1,0,0],
                                      [1,-1,1,0]]).T,
                columns=['c', 'x','coef0', 'coef1', 'coef2', 'coef3']),
                'Dormand-Prince':
                pd.DataFrame(np.array([[0,1/5,3/10,4/5,8/9,1,1],
                 [35/384,0,500/1113,125/192,-2187/6784,11/84,0],
                 [5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40],
                 [0,0,0,0,0,0,0],
                 [1/5,0,0,0,0,0,0],
                 [3/40,9/40,0,0,0,0,0],
                 [44/45,-56/15,32/9,0,0,0,0],
                 [19372/6561,-25360/2187,64448/6561,-212/729,0,0,0],
                 [9017/3168,-355/33,46732/5247,49/176,-5103/18656,0,0],
                 [35/384,0,500/1113,125/192,-2187/6784,11/84,0]],dtype=np.float64).T,
                columns=['c', 'x','xh','coef0', 'coef1', 'coef2', 'coef3',
                         'coef4','coef5','coef6']),
                'Bogacki–Shampine':
                pd.DataFrame(np.array([[0,1/2,3/4,1],
                 [2/9,1/3,4/9,0],
                 [2/9,1/3,4/9,1/8],
                 [0,0,0,0],
                 [1/2,0,0,0],
                 [0,3/4,0,0],
                 [2/9,1/3,4/9,0]]).T,
                columns=['c', 'x','xh','coef0', 'coef1', 'coef2', 'coef3']),
                'ESDIRK23':
                pd.DataFrame(np.array([[0,2-np.sqrt(2),1],
                 [1/4*np.sqrt(2),1/4*np.sqrt(2),1-1/2*np.sqrt(2)],
                 [(-5+3*np.sqrt(2))/(-12+6*np.sqrt(2)), -1/(6*(-2+np.sqrt(2))*(-1+np.sqrt(2))), (-4+3*np.sqrt(2))/(-6+6*np.sqrt(2))],
                 [0,1-1/2*np.sqrt(2), 1/4*np.sqrt(2)],
                 [0,1-1/2*np.sqrt(2),1/4*np.sqrt(2)],
                 [0,0,1-1/2*np.sqrt(2)],
                 [(4-3*np.sqrt(2))/(-6+3*np.sqrt(2)), -1/3, (-4+3*np.sqrt(2))/(-3+3*np.sqrt(2))]]).T,
                columns=['c', 'x', 'xh', 'coef0', 'coef1', 'coef2', 'd']),
                'RADAU5':
                pd.DataFrame(np.array([[(4-np.sqrt(6)/10),(4+np.sqrt(6)/10),1],
                 [(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9],
                 [(88-7*np.sqrt(6))/360, (296-169*np.sqrt(6))/1800, (-2+3*np.sqrt(6))/225],
                 [(296+169*np.sqrt(6))/1800, (88+7*np.sqrt(6))/360, (-2-3*np.sqrt(6))/225],
                 [(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9]]).T,
                columns=['c', 'x', 'coef0', 'coef1', 'coef2']),
                'AK32':
                pd.DataFrame(np.array([[0,1/4,1/2],
                                      [2/3,-4/3,5/3],
                                      [-1/2,1,1/2],
                                      [0,0,0],
                                      [1/4,0,0],
                                      [1/10,2/5,0]]).T,
                columns=['c', 'x','xh','coef0', 'coef1', 'coef2']),
    }
    order_method = {'Classic':4,
                    '3/8-rule':4,
                    'Dormand-Prince':5,
                    'Bogacki–Shampine': 3,
                    'ESDIRK23':2,
                    'RADAU5':5,
                    'AK32':3,
                   }

    N      = np.int64(round((t[1]-t[0])/dt))
    n      = len(num_methods[method]['c'])

    k      = np.zeros((x.shape[0],n),dtype=np.float64)
    absTol = 10**(-4)
    relTol = 10**(-4)

    epsTol = 0.8
    facmin = 0.1
    facmax = 5

    # categorizing the methods
    eee      = ['AK32','Dormand-Prince','Bogacki–Shampine']
    implicit = ['RADAU5']
    ESDIRK   = ['ESDIRK23']

    if (not (method in eee)) & (adap == False):
      # fixed step size RK methods
      print('Using fixed step size for: {}'.format(method))

      X    = np.zeros((x.shape[0],N),dtype=np.float64)
      X[:,0] = x
      T    = np.zeros((N),dtype=np.float64)
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
      # embedded error estimator adaptive step size methods
      print('Using Embedded error estimator for: {}'.format(method))
      p      = order_method[method]
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
              ap  = np.int64(round(N/2))
              X  = np.append(X,np.zeros((xs.shape[0],ap)),axis=1)
              T  = np.append(T,np.zeros((ap)))
              ss = np.append(ss,np.zeros((ap)))
              N = N + ap

          if j%(np.int64(round(len(X)/15)))==0:
            bs = X.nbytes/1000000
            print("{:<8} {:<8} {:<8} {:<8} \n {:>8} {:>17} {:>11} {:>15}".format('Time step:',
            																  'Step size[power]:',
            																  'Percentage:',
            																  'Array size[mb]:',
            															      round(T[j],2),
              																  round(np.log10(dt),2),
              																  round((T[j]/t[1])*100,2),
              																  bs))

          dt = np.max([facmin,np.min([(epsTol/np.float64(r))**(1/(p+1)),facmax])])*dt
          

      return T[:j],X[:,:j],ss[:j]
    elif (not (method in eee)) & (adap == True) & (not (method in ESDIRK)) & (not (method in implicit)):
      # step doubling adaptive step methods
      print('Using step doubling for: {}'.format(method))
      p      = order_method[method]
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
              ap = np.int64(round(N/2))
              X  = np.append(X,np.zeros((xs.shape[0],ap)),axis=1)
              T  = np.append(T,np.zeros((ap)))
              ss = np.append(ss,np.zeros((ap)))
              N  = N + ap

          dt = np.max([facmin,np.min([((epsTol/np.float64(r)))**(1/(p+1)),
                       facmax])])*dt
    
          if j%(np.int64(round(len(X)/15)))==0:
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
    elif ((method in implicit) & (adap==True)):
      # RADAU5
      print('Using step doubling for: {}'.format(method))
      p      = order_method[method]
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
            
          ts,xs    = rk_step_impl(fun,jac,num_methods,method,T[j],X[:,j],dt,kwargs)
          ts,x_tmp = rk_step_impl(fun,jac,num_methods,method,T[j],X[:,j],0.5*dt,kwargs)
          ts,x_tmp = rk_step_impl(fun,jac,num_methods,method,ts,x_tmp,0.5*dt,kwargs)

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

          dt = np.max([facmin,np.min([(epsTol/np.float64(r))**(1/(p+1)),
                       facmax])])*dt
    
          if j%(len(X)/100)==0:
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
    
    elif method in ESDIRK:
        # ESDIRK23
        T      = np.zeros((N))
        X      = np.zeros((x.shape[0],N))
        k      = np.zeros((x.shape[0],n))
        ss     = np.zeros((N))
        j      = 0
        X[:,0] = x
        I = np.eye(np.size(x))
        c = num_methods[method]['c']
        s = len(c)
        b = np.asarray(num_methods[method]['x'])
        d = np.asarray(num_methods[method]['d'])

        a = np.zeros([3,3])
        a[0] = [0,0,0]
        a[1] = [1-1/2*np.sqrt(2), 1-1/2*np.sqrt(2), 0]
        a[2] = [1/4*np.sqrt(2), 1/4*np.sqrt(2), 1-1/2*np.sqrt(2)]

        alpha = 0.002
        alpharef = 0.2
        epsilon = reltol

        if method == 'ESDIRK23':
            gamma = (2-np.sqrt(2))/2

        Fstage = np.zeros((x.shape[0],s+1))
        Tstage = np.zeros(s+1)
        Xstage = np.zeros((x.shape[0],s+1))
        Psistage = np.zeros((x.shape[0],s+1))
        Rstage = np.zeros((x.shape[0],s+1))
        
        Fprev = fun(T[j],X[:,j],kwargs)

        while T[j] < t[1]:
            if(T[j]+dt>t[1]):
                dt = t[1]-T[j]
    
            M = I - dt*gamma*jac(T[j],X[:,j],kwargs)
            LU, P = scipy.linalg.lu_factor(M)
            
            AcceptStep = False
            while not AcceptStep:
                Tstage[1] = T[j]
                Xstage[:,1] = X[:,j]
                Fstage[:,1] = Fprev
                
                for i in range(2,s+1):
                    Psistage[:,i] = Xstage[:,1] + dt*  np.sum(np.array([a[i-1,z]*Fstage[:,z] for z in range(1,i)]), axis=0)
                    Tstage[i] = T[j] + c[i-1]*dt
                    Xstage[:,i] = X[:,j] + c[i-1]*dt*Fstage[:,1]
                    Fstage[:,i] = fun(Tstage[i],Xstage[:,i],kwargs)
                    Rstage[:,i] = Xstage[:,i] - dt*gamma*Fstage[:,i] - Psistage[:,i]
                    
                    rnewt = np.max(np.abs(Rstage[:,i])/(absTol + np.abs(Xstage[:,i])*relTol))
                    diverging = False
                    
                    prevXs = np.zeros([len(Xstage[:,i]), 3])
                    prevXs[:,0] = Xstage[:,i]
                    while (rnewt > (1) & (not diverging)):
                        Dx = scipy.linalg.lu_solve((LU, P),Rstage[:,i])

                        rprev = rnewt

                        Xstage[:,i] = Xstage[:,i] - Dx
                        Fstage[:,i] = fun(Tstage[i],Xstage[:,i],kwargs)
                        Rstage[:,i] = Xstage[:,i] - dt*gamma*Fstage[:,i] - Psistage[:,i]
                        
                        prevXs = np.roll(prevXs,1)
                        prevXs[:,0] = Xstage[:,i]
                        
                        rnewt = np.max(np.abs(Rstage[:,i])/(absTol + np.abs(Xstage[:,i])*relTol))
                        
                        alpha = np.max([alpha, rnewt/rprev])
                        if (alpha >= 1):
                            diverging = True
                    dtalpha = alpharef/alpha*dt
                
                e = np.abs(dt* np.array([d[z-i]*Fstage[:,z] for z in range(1,s)]))
                num = absTol + np.abs(Xstage[:,s-1])*relTol
                r   = np.max(e/num)
                AcceptStep = (r <= 1) & (not diverging)
                
                if AcceptStep:
                    X[:,j+1] = Xstage[:,s]
                    T[j+1]   = T[j] + dt 
                    ss[j+1]  = dt
                    Fprev = Fstage[:,s]
                    j+=1
                    
                    if j+1==N:
                      ap  = np.int32(round(N/2))
                      X  = np.append(X,np.zeros((Xstage.shape[0],ap)),axis=1)
                      T  = np.append(T,np.zeros((ap)))
                      ss = np.append(ss,np.zeros((ap)))
                      N = N + ap
                      
                dt = (epsTol/r)**(1/3)*dt
                dt = np.min([dt,dtalpha])
        
            if j%(len(X)/100)==0:
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
    else:
      print('Parameters not specified correctly')

#Runge_Kutta(VanDerPol,np.array([0.5,0.5]),[0,25],0.01,10,method='ESDIRK23',adap=True,jac=JacVanDerPol)


def tf(t,x,mu):
  return -x[0]

def true_tf(t):
  return np.exp(-t)




dt = 10**(-2)
lamda = -1
x0  = np.array([1])
ti = [0,20]


r = ode(testEqn,
        JacTestEqn).set_integrator('vode',
                                      method='bdf',
                                      with_jacobian=True,
                                      order=15)

r.set_initial_value(x0, ti[0]).set_f_params([lamda]).set_jac_params([lamda])
x_sci_s = []
t       = np.arange(0,ti[1],0.01)
# Solving using the scipy solver
while r.successful() and r.t < ti[1]:
    xn = r.integrate(r.t+0.01)
    x_sci_s.append(xn[0])


T_AK32,X_AK32 = Runge_Kutta(testEqn,
                                      x0,
                                      ti,
                                      dt,
                                      [lamda],
                                      method='Classic',
                                      adap=False,
                                      jac=JacTestEqn)

fig, ax = plt.subplots(1, 1, figsize=(15,10), sharex=False)

ax.plot(T_AK32,X_AK32[0,:],label='ERK4')
ax.plot(t,x_sci_s,'--',label='Scipy 15')

ax.set_title('Solution to the test problem - Classic RK4 - fixed step size')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.legend()
fig.savefig('2-1-3.eps', format='eps', dpi=500, bbox_inches='tight')


T_AK32,X_AK32,SS = Runge_Kutta(testEqn,
                                      x0,
                                      ti,
                                      dt,
                                      [lamda],
                                      method='Classic',
                                      adap=True)

fig, ax = plt.subplots(1, 1, figsize=(15,10), sharex=False)

ax.plot(T_AK32,X_AK32[0,:],label='ERK4')
ax.plot(t,x_sci_s,'--',label='Scipy 15')

ax.set_title('Solution to the test problem - Classic RK4 - adaptive step size')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.legend()
fig.savefig('2-2-3.eps', format='eps', dpi=500, bbox_inches='tight')




T_AK32,X_AK32,SS = Runge_Kutta(testEqn,
                                      x0,
                                      ti,
                                      dt,
                                      [lamda],
                                      method='Dormand-Prince',
                                      adap=True,
                                      jac=JacTestEqn)

fig, ax = plt.subplots(1, 1, figsize=(15,10), sharex=False)

ax.plot(T_AK32,X_AK32[0,:],label='DOPRI')
ax.plot(t,x_sci_s,'--',label='Scipy 15')

ax.set_title('Solution to the test problem - Dormand-Prince - adaptive step size')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.legend()
fig.savefig('2-3-3.eps', format='eps', dpi=500, bbox_inches='tight')






dt = np.linspace(1e-3,1e-1,num=200)
sol = np.zeros(np.size(dt))
x0 = np.array([1,1])
for i in range(len(dt)):
  t,x,ss = Runge_Kutta(tf,
                    x0,
                    [0,1],
                    dt[i],
                    -1,
                    method='Classic',adap=True)

  sol[i] = np.amax(np.abs(x[0]-true_tf(t)))

plt.plot(np.log(dt),np.log(sol),label='LTE RK')

plt.show()


abstol = 10**(-6)
reltol = 10**(-6)
x0 = np.array([0.5,0.5])

dt = 10**(-2)
mu = 30
ti  = [0,10]

mu = 1
ti  = [0,50]



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


plt.plot(t, x_sci_s[0], t ,x_sci_s[1])
#plt.show()
#plt.plot(x_sci_s[0],x_sci_s[1])
plt.show()

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
                          10**(-5),
                          mu,
                          method='Dormand-Prince')

T_BS_3,X_BS_3,SS_BS_3 = Runge_Kutta(VanDerPol,
                          x0,
                          ti,
                          dt,
                          mu,
                          method='Bogacki–Shampine')




#------------------------------ PLOTTING MADNESS BEGIN

dt = 10**(-2)
lamda = -1
x0  = np.array([1])
ti = [0,20]

T_AK32,X_AK32,SS_AK32 = Runge_Kutta(testEqn,
                                      x0,
                                      ti,
                                      dt,
                                      [lamda],
                                      method='AK32',
                                      adap=True,
                                      jac=JacTestEqn)

fig, ax = plt.subplots(2, 1, figsize=(15,10), sharex=False)

ax[0].plot(T_AK32,X_AK32[0,:],label='Developed RK scheme')
ax[0].set_title('Solution to the test problem - Developed RK scheme')
ax[0].set_xticks([])
ax[0].set_ylabel('x')

ax[1].plot(T_AK32,SS_AK32,label='Step Size')
ax[1].set_title('Step size')
ax[1].set_xlabel('t')
ax[1].set_ylabel('dt')
plt.show()

fig.savefig('3-1-4.eps', format='eps', dpi=500, bbox_inches='tight')


x0 = np.array([0.5,0.5])

abstol = 10**(-4)
reltol = 10**(-4)

dt = 10**(-2)
mu = 3
ti  = [0,5*mu]

T_AK32,X_AK32,SS_AK32 = Runge_Kutta(VanDerPol,
                                      x0,
                                      ti,
                                      dt,
                                      mu,
                                      method='AK32',
                                      adap=True,
                                      jac=JacVanDerPol)


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


fig, ax = plt.subplots(2, 1, figsize=(15,10), sharex=False)
# Plotting the results
ax[0].plot(t[:len(x_sci_s[0][:])],x_sci_s[0][:],'-o',label='Scipy')
#ax[0].plot(T_C_3,X_C_3[0,:],label='RK4 FS')
ax[0].plot(T_AK32,X_AK32[0,:],'-o',label='RK4 AS')
#ax[0].plot(T_DP_3,X_DP_3[0,:],'-o',label='DP54 AS')
#ax[0].plot(T_BS_3,X_BS_3[0,:],label='BS AS')
ax[0].set_title(r'Phase one of the Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(dt,mu,abstol,reltol))
ax[0].set_xticks([])
ax[0].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)

ax[1].plot(t[:len(x_sci_s[1][:])],x_sci_s[1][:],'-o',label='Scipy')
#ax[1].plot(T_C_3,X_C_3[1,:],label='RK4 FS')
ax[1].plot(T_AK32,X_AK32[1,:],'-o',label='RK4 AS')
#ax[1].plot(T_DP_3,X_DP_3[1,:],'-o',label='DP54 AS')
#ax[1].plot(T_BS_3,X_BS_3[1,:],label='BS AS')
ax[1].set_title(r'Phase two of the Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(dt,mu,abstol,reltol))
ax[1].set_xticks([])
ax[1].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)

plt.show()

fig.savefig('3-1-7.eps', format='eps', dpi=500, bbox_inches='tight')



abstol = 10**(-4)
reltol = 10**(-4)
x0 = np.array([0.5,0.5])

dt = 10**(-2)
mu = 10
ti  = [0,5*mu]



r = ode(VanDerPol,
        JacVanDerPol).set_integrator('vode',
                                      method='bdf',
                                      with_jacobian=True,
                                      order=45)
r.set_initial_value(x0, ti[0]).set_f_params(mu).set_jac_params(mu)
x_sci_s = [[],[]]
t       = np.arange(0,ti[1]+0.01,0.01)
# Solving using the scipy solver
while r.successful() and r.t < ti[1]:
    xn = r.integrate(r.t+0.01)
    x_sci_s[0].append(xn[0])
    x_sci_s[1].append(xn[1])

T_C_A3,X_C_A3,SS_C_A3 = Runge_Kutta(VanDerPol,
                          x0,
                          ti,
                          dt,
                          mu,
                          method='Classic',
                          adap=True)

fig, ax = plt.subplots(3, 1, figsize=(15,10), sharex=False)
# Plotting the results
ax[0].plot(t[:len(x_sci_s[0][:])],x_sci_s[0][:],'-o',label='Scipy')
#ax[0].plot(T_C_3,X_C_3[0,:],label='RK4 FS')
ax[0].plot(T_C_A3,X_C_A3[0,:],'-o',label='RK4 AS')
#ax[0].plot(T_DP_3,X_DP_3[0,:],'-o',label='DP54 AS')
#ax[0].plot(T_BS_3,X_BS_3[0,:],label='BS AS')
ax[0].set_title(r'Phase one of the Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(dt,mu,abstol,reltol))
ax[0].set_xticks([])
ax[0].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)

ax[1].plot(t[:len(x_sci_s[1][:])],x_sci_s[1][:],'-o',label='Scipy')
#ax[1].plot(T_C_3,X_C_3[1,:],label='RK4 FS')
ax[1].plot(T_C_A3,X_C_A3[1,:],'-o',label='RK4 AS')
#ax[1].plot(T_DP_3,X_DP_3[1,:],'-o',label='DP54 AS')
#ax[1].plot(T_BS_3,X_BS_3[1,:],label='BS AS')
ax[1].set_title(r'Phase two of the Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(dt,mu,abstol,reltol))
ax[1].set_xticks([])
ax[1].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)

ax[2].plot(T_C_A3[1:],np.log(SS_C_A3[1:]),'-o',label='SS RK4')
#ax[2].plot(T_DP_3[1:],np.log(SS_DP_3[1:]),label='SS DP54')
#ax[2].plot(T_BS_3[1:],np.log(SS_BS_3[1:]),label='SS BS')
ax[2].set_title('Semi log-plot of step sizes with tolerance {}'.format(10**(-4)))
ax[2].legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)
plt.savefig('./figs/sol_VDP_CRKA_mu10.pdf')
plt.show()

plt.figure()
plt.plot(x_sci_s[0][:],x_sci_s[1][:],'-o',label='Scipy')
#plt.plot(X_C_3[0,:],X_C_3[1,:],label='RK4 FS')
plt.plot(X_C_A3[0,:],X_C_A3[1,:],'-o',label='RK4 AS')
#plt.plot(X_DP_3[0,:],X_DP_3[1,:],'-o',label='DP54 AS')
#plt.plot(X_BS_3[0,:],X_BS_3[1,:],label='BS AS')
plt.title(r'Phase state of the Van Der Pol. [SS: {}, $\mu = {}$, abstol = {}, reltol = {}]'.format(dt,mu,abstol,reltol))
plt.legend(bbox_to_anchor=(-0.15, 1), loc=2, borderaxespad=0.)
plt.savefig('./figs/sol_VDP_CRKA_PS_mu10.pdf')
plt.show()


abstol = 10**(-4)
reltol = 10**(-4)
x0 = np.array([0.5,0.5])

dt = 10**(-2)

mu = 3
ti  = [0,25]

T_RADAU,X_RADAU,SS_RADAU = Runge_Kutta(VanDerPol,
                                      x0,
                                      ti,
                                      dt,
                                      mu,
                                      method='RADAU5',
                                      adap=True,
                                      jac=JacVanDerPol)
T_ESDIRK,X_ESDIRK,SS_ESDIRK = Runge_Kutta(VanDerPol,
                                          x0,
                                          ti,
                                          dt,
                                          mu,
                                          method='ESDIRK23',
                                          adap=True,
                                          jac=JacVanDerPol)


mu = 10
ti  = [0,25]


T_RADAU2,X_RADAU2,SS_RADAU2 = Runge_Kutta(VanDerPol,
                                      x0,
                                      ti,
                                      dt,
                                      mu,
                                      method='RADAU5',
                                      adap=True,
                                      jac=JacVanDerPol)
T_ESDIRK2,X_ESDIRK2,SS_ESDIRK2 = Runge_Kutta(VanDerPol,
                                          x0,
                                          ti,
                                          dt,
                                          mu,
                                          method='ESDIRK23',
                                          adap=True,
                                          jac=JacVanDerPol)


mu = 3
ti  = [0,25]

fig, ax = plt.subplots(3, 1, figsize=(15,10), sharex=False)

ax[0].plot(T_ESDIRK, X_ESDIRK[0,:], label='X_1')
ax[0].set_title(r'X1 variable of the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))
ax[0].set_xticks([])
ax[1].plot(T_ESDIRK, X_ESDIRK[1,:], 'orange', label='X_2')
ax[1].set_title(r'X2 variable of the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))
ax[1].set_xticks([])
ax[2].plot(T_ESDIRK, SS_ESDIRK, 'red', label='StepSize')
ax[2].set_title(r'Step size for the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))

fig.savefig('ESDIRK1.eps', format='eps', dpi=500, bbox_inches='tight')

mu = 10

fig, ax = plt.subplots(3, 1, figsize=(15,10), sharex=False)

ax[0].plot(T_ESDIRK2, X_ESDIRK2[0,:], label='X_1')
ax[0].set_title(r'X1 variable of the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))
ax[0].set_xticks([])
ax[1].plot(T_ESDIRK2, X_ESDIRK2[1,:], 'orange', label='X_2')
ax[1].set_title(r'X2 variable of the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))
ax[1].set_xticks([])
ax[2].plot(T_ESDIRK2, SS_ESDIRK2, 'red', label='StepSize')
ax[2].set_title(r'Step size for the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))

fig.savefig('ESDIRK2.eps', format='eps', dpi=500, bbox_inches='tight')


mu = 3

fig, ax = plt.subplots(3, 1, figsize=(15,10), sharex=False)

ax[0].plot(T_RADAU, X_RADAU[0,:], label='X_1')
ax[0].set_title(r'X1 variable of the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))
ax[0].set_xticks([])
ax[1].plot(T_RADAU, X_RADAU[1,:], 'orange', label='X_2')
ax[1].set_title(r'X2 variable of the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))
ax[1].set_xticks([])
ax[2].plot(T_RADAU, SS_RADAU, 'red', label='StepSize')
ax[2].set_title(r'Step size for the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))

fig.savefig('RADAU1.eps', format='eps', dpi=500, bbox_inches='tight')


mu = 10

fig, ax = plt.subplots(3, 1, figsize=(15,10), sharex=False)

ax[0].plot(T_RADAU2, X_RADAU2[0,:], label='X_1')
ax[0].set_title(r'X1 variable of the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))
ax[0].set_xticks([])
ax[1].plot(T_RADAU2, X_RADAU2[1,:], 'orange', label='X_2')
ax[1].set_title(r'X2 variable of the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))
ax[1].set_xticks([])
ax[2].plot(T_RADAU2, SS_RADAU2, 'red', label='StepSize')
ax[2].set_title(r'Step size for the Van Der Pol. [$\mu = {}$, abstol = {}, reltol = {}]'.format(mu,abstol,reltol))

fig.savefig('RADAU2.eps', format='eps', dpi=500, bbox_inches='tight')

#cProfile.run("Runge_Kutta(VanDerPol,np.array([0.5,0.5]),[0,10],10**(-4),mu,method='Bogacki–Shampine',adap=True)",sort='cumtime')


#cProfile.run("Runge_Kutta(VanDerPol,np.array([0.5,0.5]),[0,25],0.01,10,method='Classic',adap=True,jac=JacVanDerPol)",sort='cumtime')

#Radau5 : fcalls 12120
#jcalls 690
#steps 126

#ESDIRK23 : fcalls 162567
#jcalls 81273
#steps 81273

#ERK4 : fcalls 9840
#steps: 413
