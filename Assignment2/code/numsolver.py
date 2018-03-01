# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

class numerical_solvers(object):
        
    def NewtonsMethod(self,ResidualFunJac, x0, tol, maxit, varargin):
        self.k = 0
        self.x = x0
        self.R,self.dRdx = ResidualFunJac(self.x,varargin)
        while (self.k < maxit) and (np.linalg.norm(self.R,np.inf) > tol):
            self.k += 1
            self.dx = np.linalg.solve(self.dRdx,self.R)
            self.x = self.x - self.dx
            self.R,self.dRdx = ResidualFunJac(self.t,self.x,varargin)
        return self.x   
    
    def NewtonsMethodODE(self,FunJac, tk, xk, dt, xinit, tol, maxit, kwargs):
        self.k = 0
        self.t = tk + dt
        self.x = xinit
        self.f,self.J = FunJac(self.t,self.x,kwargs)
        self.R = self.x - self.f*dt - xk
        self.I = np.eye(np.size(xk))
        while (self.k < maxit) and (np.linalg.norm(self.R,np.inf) > tol):
            self.k += 1
            self.dRdx = self.I - self.J*dt
            self.dx = np.linalg.solve(self.dRdx,np.transpose(self.R))
#            self.dx = self.dx[0]
            self.x = self.x - self.dx
            self.f,self.J = FunJac(self.t,self.x,kwargs)
            self.R = self.x - self.f*dt - xk
        return self.x
    
    def VanDerPol(self,t,x,mu):
        # VANDERPOL Implementation of the Van der Pol model
        #
        # Syntax: xdot = VanDerPol(t,x,mu)
        self.mu=mu
        self.t=t
        self.x=x
        self.xdot = np.zeros([2, 1])
        self.xdot[0] = self.x[1]
        self.xdot[1] = self.mu*(1-self.x[0]*self.x[0])*self.x[1]-self.x[0]
        return self.xdot.T[0]

    def JacVanDerPol(self,t,x,mu):
        # JACVANDERPOL Jacobian for the Van der Pol Equation
        #
        # Syntax: Jac = JacVanDerPol(t,x,mu)
        self.x =x
        self.mu=mu
        self.t=t
        self.Jac = np.zeros([2, 2])
        self.Jac[1,0] = -2*self.mu*self.x[0]*self.x[1]-1.0
        self.Jac[0,1] = 1.0
        self.Jac[1,1] = self.mu*(1-self.x[0]*self.x[0])
        return self.Jac

    def VanderPolfunjac(self,t,x,mu):
        return self.VanDerPol(t,x,mu), self.JacVanDerPol(t,x,mu)
    
    def ImplicitEulerFixedStepSize(self,funJac,ta,tb,N,xa,kwargs=None):

    # Compute step size and allocate memory
        self.dt = (tb-ta)/N;
        self.nx = np.size(xa);
        self.X  = np.zeros([N+1,self.nx]);
        self.T  = np.zeros([N+1,1]);
    
        self.tol = 10**(-8);
        self.maxit = 100;

    #Eulers Implicit Method
        self.T[0,:] = ta;
        self.X[0,:] = xa;
        for k in range(N):
            self.f,self.j = funJac(self.T[k],self.X[k,:],kwargs);
#            self.f=self.f[0]
            self.T[k+1,:] = self.T[k,:] + self.dt;
            self.xinit = self.X[k,:] + np.transpose(self.f*self.dt);
            self.X[k+1,:] = self.NewtonsMethodODE(funJac,
                                        self.T[k,:],
                                        self.X[k,:],
                                        self.dt,
                                        self.xinit,
                                        self.tol,
                                        self.maxit,
                                        kwargs);
         
        return self.T,self.X
   
    def ImplicitEulerAdaptiveStepSize(self,fun,ta,tb,xa,h,absTol,relTol,
                                      epstol,kwargs=None):
        t=ta
        self.h = h
        self.x = xa
        self.facmin = 0.1
        self.facmax = 1
        self.T = np.array([t])
        self.X = np.array([self.x])
        self.ss =np.array([self.h])
        while t < tb:
            if(self.t+self.h>tb):
                self.h = tb-t
                
            self.f,self.na = fun(t,xa,kwargs)
            print(self.f)
            self.AcceptStep = False
            while not self.AcceptStep:
                self.x1 = self.x + np.transpose(self.h*self.f)
                #print(self.x1)
                self.x1 = self.NewtonsMethodODE(fun,
                                                t,
                                                self.x1,
                                                self.h,
                                                xa.T,
                                                absTol,
                                                100,
                                                kwargs)
                
                self.hm = 0.5*self.h
                self.tm = t + self.hm
                self.xm = self.x + np.transpose(self.hm*self.f)
                self.x1hat = self.NewtonsMethodODE(fun,
                                                   t,
                                                   self.x1,
                                                   self.h,
                                                   xa.T,
                                                   absTol,
                                                   100,
                                                   kwargs)
                
                self.e = self.x1hat - self.x1
                print(self.e)
                self.r = np.max(np.abs(self.e)/np.max([absTol,np.amax(self.x1hat*relTol)]))
    
                self.AcceptStep = (self.r <= 1)
                print("Timestep: {} and r value: {}".format(t,self.r))
                if self.AcceptStep:
                    t =t+self.h
                    self.x = self.x1hat
                    
                    self.T = np.append(self.T,t)
                    self.X = np.append(self.X,self.x)
                    self.ss =np.append(self.ss,self.h)
                    
                self.h = np.max([self.facmin,
                                np.sqrt(epstol)/self.r*self.facmax])*self.h
                                
        return self.T,self.X,self.h
                                
                
            
        
        
        
        









