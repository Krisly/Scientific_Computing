import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.sparse.linalg import spsolve
from scipy import signal
from scipy import misc


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

print(poisson5(3).todense())

def rhs_fun(f,x,y,g,mod=0):
	'''
	This calculates the right hand-side used to solved the system Au=f for the poisson equation in a
	2D square domain, it is for the function "solve_sys". 
	'''
	m = x.shape[0] -2
	h = 1/(x.shape[0]+1)

	elems = f(x,y)
	he = np.array([[0,1,0],[1,-4,1],[0,1,0]])
	rhs = elems - (1/(6*h**2))*g 
	return rhs.flatten()

def solve_sys(f,x,y,g,m,method='5-point'):
	'''
	This function solves the a 2D problem with a square domain [a ; b] x [c ; d], using
	either the 5-point Laplacian or 9-point Laplacian, with specified parameters:

	fun    : The function which is being solved

	u      : The bundary conditions

	m      : The amount of grid-points. The realtioship of m to the step-size is h = 1/(m+1)

	ep     : This is the endpoints of the 2D-domain

	method : This specify whether to use the 5-point or 9-point Laplacian
	'''

	if method == '5-point':
		return spsolve(poisson5(m), rhs_fun(f,y,y,g))
	elif method == '9-point':
		return spsolve(poisson9(m), rhs_fun(f,x,y,g))

def u_excact_0(x,y):
	# Test function used to evaluate convergence rate

	return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)

def lap_u_excact_0(x,y):
	# Test function used to evaluate convergence rate

	return -16*np.pi**2*(2*np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)*(x**2+y**2))

def u_excact_1(x,y):
	# Test function used to evaluate convergence rate

	return x**2+y**2

def lap_u_excact_1(x,y):

	return 0*np.ones(x.shape)
	
def u_excact_2(x,y):
	# Test function used to evaluate convergence rate

	return np.sin(2*np.pi*np.abs(x-y)**(2.5))
