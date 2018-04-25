import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.sparse.linalg import spsolve
from scipy import signal
from scipy import misc
from numpy.linalg import inv

import time
from helper_functions import plot_pois,u_excact_0,u_excact_1,lap_u_excact_0,err_plot
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import cg


def poisson5(m):
	# Calculates the 5-point Laplacian system matrix A
	e = np.repeat(1,m)
	S = sp.sparse.spdiags(np.array([e,-2*e,e]),np.array([-1,0,1]),m,m)
	I = sp.sparse.eye(m)
	A = sp.sparse.kron(I,S) + sp.sparse.kron(S,I)
	A =(m+1)**2*A
	return A

def poisson9(m):
	# Calculates the 9-point Laplacian system matrix A
	e = np.repeat(1,m)
	S = sp.sparse.spdiags(np.array([-e, -10*e ,-e]), np.array([-1, 0, 1]), m, m)
	I = sp.sparse.spdiags(np.array([-1/2*e, e ,-1/2*e]), np.array([-1, 0 ,1]), m, m)
	A = 1/6*(m+1)**2*(sp.sparse.kron(I,S)+sp.sparse.kron(S,I))
	return A

def form_rhs(f,x,y,g):
    h = 1/(x.shape[0]+1)
    se = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    fun = f(x,y)
    fun_lp5 = (h**2/12)*scipy.signal.convolve2d(fun,se,mode='same')
    rhs = (fun+fun_lp5) - g
    return rhs.flatten()

# Solving the system for the test function 2
m     = 50

# Structuring elements
lp9          = np.array([[1,4,1],[4,-20,4],[1,4,1]]) 													# 9-point Laplace structuring element

# Domain Discritization
h            = 1/(m+1)				   				 													# Step size
x            = np.linspace(0, 1, m+2)																	# X range
y            = np.linspace(0, 1, m+2)				 													# Y range
X, Y         = np.meshgrid(x, y)					 													# Domain meshgrid

# Boundary conditions
g            = u_excact_0(X,Y)                      													# Full rhs
g[1:-1,1:-1] = 0																						# Setting  boundary condition by convulution
g            = 1/(6*h**2)*signal.convolve2d(g,lp9,mode='valid')

# Solving system
u            = spsolve(poisson9(m), form_rhs(lap_u_excact_0,X[1:-1,1:-1],Y[1:-1,1:-1],g))
plot_pois(X,Y,u.reshape(m,m),u_excact_0)



ss  = np.arange(10,200,10)
sol = np.zeros((len(ss)))

for i in range(len(ss)):

    # Domain Discritization
    m        = ss[i]								 												# Number of grid points
    lp9          = np.array([[1,4,1],[4,-20,4],[1,4,1]]) 													# 9-point Laplace structuring element

    # Domain Discritization
    h            = 1/(m+1)				   				 													# Step size
    x            = np.linspace(0, 1, m+2)																	# X range
    y            = np.linspace(0, 1, m+2)				 													# Y range
    X, Y         = np.meshgrid(x, y)					 													# Domain meshgrid

    # Boundary conditions
    g            = u_excact_0(X,Y)                      													# Full rhs
    g[1:-1,1:-1] = 0																						# Setting  boundary condition by convulution
    g            = 1/(6*h**2)*signal.convolve2d(g,lp9,mode='valid')

    # Solving system
    u            = spsolve(poisson9(m), form_rhs(lap_u_excact_0,X[1:-1,1:-1],Y[1:-1,1:-1],g))
    sol[i]       = np.max(np.abs(u.reshape(m,m)-u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1])))					# Error calculation

# Error plot
h = np.array([1/(x+1) for x in ss])                                                                     # Step size vector
err_plot(h,sol,s_order=2,e_order=4)                                                                    # Creating the error plot
#plot_pois(X,Y,u,u_excact_0)
