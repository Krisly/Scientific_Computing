# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def func_counter(func_counter=0):
    func_counter += 1
    return func_counter

class numerical_solvers(object):
    t      = np.array([0,100])
    x0     = np.array([0,0])
    absTol = 10**(-6)
    relTol = 10**(-3)
    epsTol = 0.8
    maxit  = 100
    R      = 0
    k      = 0
    param  = 0
    dt     = 0.001
    facmin = 0.1
    facmax = 5
    seed   = 3
    
    def NewtonsMethod(self,rjf,x0):
        k        = self.k
        x        = x0
        t        = self.t
        res,dRdx = rjf(t,x,self.param)
        
        while (k < self.maxit) and (np.linalg.norm(res,np.inf) >
                                         self.absTol):
            k        += 1
            dx       = np.linalg.solve(dRdx,res)
            x        = x - dx
            res,dRdx = rjf(t,x,self.param)
        return x   
    
    def NewtonsMethodODE(self,rjf, tk,x, xk,dt):
        k   = self.k
        t   = tk + dt
        f,J = rjf(t,x,self.param)
        res = x - f*dt - xk
        I   = np.eye(np.size(xk))
        
        while (k < self.maxit) and (np.linalg.norm(res,np.inf) > self.absTol):
            k    += 1
            dRdx = I - J*dt
            dx   = np.linalg.solve(dRdx,np.transpose(res))
            x    = x - dx
            f,J  = rjf(t,x,self.param)
            res  = x - f*dt - xk
        return x
    
    def VanDerPol(self,t,x,mu):
        xdot    = np.zeros([2, 1])
        xdot[0] = x[1]
        xdot[1] = mu*(1-x[0]*x[0])*x[1]-x[0]
        return xdot.T[0]

    def JacVanDerPol(self,t,x,mu):
        Jac      = np.zeros([2, 2])
        Jac[1,0] = -2*mu*x[0]*x[1]-1.0
        Jac[0,1] = 1.0
        Jac[1,1] = mu*(1-x[0]*x[0])
        return Jac

    def VanderPolfunjac(self,t,x,mu):
        return self.VanDerPol(t,x,mu),self.JacVanDerPol(t,x,mu)
    
    def PreyPredator(self, t,x,params):
        # PreyPredator Implementation of the PreyPredator model
        #
        # Syntax: xdot = VanDerPol(t,x,params)
        a = params[0]
        b = params[1]
        xdot = np.zeros([2, 1])
        xdot[0] = a*(1-x[1])*x[0]
        xdot[1] = -b*(1-x[0])*x[1]
        return xdot.T[0]
    
    def JacPreyPredator(self, t,x,params):
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
        return [self.PreyPredator(t,x,params), self.JacPreyPredator(t,x,params)]
    
    
    def ImplicitEulerFixedStepSize(self,rjf):
        dt     = self.dt
        N      = np.int32(round((self.t[1]-self.t[0])/self.dt))
        nx     = np.size(self.x0);
        X      = np.zeros([N+1,nx]);
        T      = np.zeros([N+1,1]);
        T[0,:] = self.t[0];
        X[0,:] = self.x0;
        k      = self.k
        
        for k in range(N):
            f,j      = rjf(T[k],X[k,:],self.param);
            T[k+1,:] = T[k,:] + dt;
            X[k+1,:] = self.NewtonsMethodODE(rjf,T[k,:],X[k,:],X[k,:],dt);
        return T,X
   
    def ImplicitEulerAdaptiveStepSize(self,fun,absTol_as = None):

        iea_t        = self.t[0]
        tend         = self.t[1]
        x1           = self.x0
        dt           = self.dt
        T            = np.zeros([1,],dtype=np.float64)
        T[0]         = iea_t
        X            = np.array([x1[0],x1[1]], dtype=np.float64)
        ss           = np.zeros([1,],dtype=np.float64)
        ss[0]        = self.dt
        accept       = 0
        nran         = 0
        nreject      = 0
        t1           = iea_t
        
        while t1 < tend:
            if(iea_t+dt>tend):
                dt = tend-iea_t

            AcceptStep = False

            while not AcceptStep:    
                iea_t =iea_t+dt
                x1    = self.NewtonsMethodODE(fun,iea_t,x1,x1,dt)
                
                hm = 0.5*dt
                tm = t1 + hm
                x1hat = self.NewtonsMethodODE(fun,tm,x1,x1,dt)
                
                x1hat = self.NewtonsMethodODE(fun,iea_t,x1hat,x1hat,dt)
    
                e = x1hat - x1
                if absTol_as!=None:
                    r = np.max(np.abs(e)/np.max([absTol_as,
                           np.max(np.abs(x1hat)*self.relTol)]))
                else:
                    r = np.max(np.abs(e)/np.max([self.absTol,
                           np.max(np.abs(x1hat)*self.relTol)]))
    
                AcceptStep = (r <= 1)
                
                if AcceptStep:
                    x  = x1hat
                    T  = np.append(T,iea_t)
                    X  = np.append(X,x)
                    ss = np.append(ss,dt)
                    accept  = func_counter(accept)
                    t1 = iea_t
                
                dt = np.max([self.facmin,
                          np.min([np.sqrt(self.epsTol/np.float64(r)),
                                          self.facmax])])*dt
    
                nran = func_counter(nran)
                
        nreject = nran - accept   
        return T,X.reshape((len(T),2)),ss,nreject
 
    def ScalarStdWienerProcess(self,T,N,Ns):
        np.random.seed(self.seed)
        dt  = T/N
        dW  = np.sqrt(dt)*np.random.randn(Ns,N)
        W   = np.append(np.zeros([Ns,1]), np.cumsum(dW,1), axis=1)
        Tw  = np.arange(0,T+dt/2,dt)
        return W,Tw,dW


    def SDEeulerExplicitExplicit(self,ffun,gfun,T,x0,W):
        N      = np.size(T)-1
        nx     = np.size(x0)
        X      = np.zeros([nx,N+1])
        X[:,0] = x0
        for k in range(0,N):
            dt       = T[k+1]-T[k]
            dW       = W[k+1]-W[k]
            f        = ffun(T[k],X[:,k],self.param)
            g        = gfun(T[k],X[:,k],self.param)
            X[:,k+1] = X[:,k] + f*dt + g*dW
        return X

    def SDEeulerImplicitExplicit(self,ffun,gfun,T,x0,dW):
        N      = np.size(T)-1
        nx     = np.size(x0)
        X      = np.zeros([nx,N+1])
        X[:,0] = x0
        for k in range(0,N):
            dt       = T[k+1]-T[k]
            f        = ffun(T[k],X[:,k],self.param)
            g        = gfun(T[k],X[:,k],self.param)
            psi      = X[:,k] + g*dW[:,k]
            X        = psi + f*dt
            #X[:,k+1] = self.NewtonsMethodNewton(ffun+gfun,X)
            X[:,k+1] = self.NewtonsMethodODE()
        return X

    def LangevinDrift(self,t,x,p):
        x     = x
        lamda = p[0]
        f     = lamda*x
        return f

    def LangevinDiffusion(self,t,x,p):
        sigma = p[1]
        g     = sigma
        return g

                
            
        
        
        
        









