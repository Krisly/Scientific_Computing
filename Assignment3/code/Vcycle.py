#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 21:20:12 2018

@author: carl
"""
from helper_functions import *

# exact solution and RHS
u=lambda x,y:  np.exp(np.pi*x)*np.sin(np.pi*y)+0.5*(x*y)**2
f=lambda x,y: x**2+y**2
m=2**6-1
U =np.zeros(m*m,1)
F =form_rhs(0,1,m,f,0,u) ## TODO: Form the right-hand side
epsilon = 1.0E-10
for i in range(1,100):
    R =F+Amult(U,m)
    print('*** Outer iteration: #3d, rel. resid.: #e\n', i, norm(R,2)/norm(F,2))
    if(np.norm(R,2)/np.norm(F,2) < epsilon):
        break
    U=Vcycle(U,omega,3,m,F)

def Vcycle(U,omega,nsmooth,m,F):
    # Approximately solve: A*U = F
    h=1.0/(m+1)
    l2m=np.log2(m+1)
    #assert(l2m==round(l2m))
    #assert(length(U)==m*m)
    if(m==1):
        # if we are at the coarsest level
        # TODO: solve the only remaining equation directly!
    else:
        # 1. TODO: pre-smooth the error
        #    perform <nsmooth> Jacobi iterations
        # 2. TODO: calculate the residual
        # 3. TODO: coarsen the residual
        # 4. recurse to Vcycle on a coarser grid
        mc=(m-1)/2
        Ecoarse=Vcycle(np.zeros(mc*mc,1),omega,nsmooth,mc,-Rcoarse)
        # 5. TODO: interpolate the error
        # 6. TODO: update the solution given the interpolated error
        # 7. TODO: post-smooth the error
        #    perform <nsmooth> Jacobi iterations
    return Unew
