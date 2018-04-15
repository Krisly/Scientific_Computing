import numpy as np
import scipy as sp
import scipy.sparse

def poisson5(m):
	e = np.repeat(1,m)
	S = sp.sparse.spdiags(np.array([e,-2*e,e]),np.array([-1,0,1]),m,m)
	I = sp.sparse.eye(m)
	A = sp.sparse.kron(I,S)+ sp.sparse.kron(S,I)
	A =(m+1)**2*A
	return A

def poisson9(m):
	e = np.repeat(1,m)
	S = sp.sparse.spdiags(np.array([-e, -10*e ,-e]), np.array([-1, 0, 1]), m, m)
	I = sp.sparse.spdiags(np.array([-1/2*e, e ,-1/2*e]), np.array([-1, 0 ,1]), m, m)
	A = 1/6*(m+1)**2*(sp.sparse.kron(I,S)+sp.sparse.kron(S,I))
	return A

def rhs_fun(fun,x,y):
	rhs = fun(x,y)
	return rhs

def func(x,y):
	return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)

xv, yv = np.meshgrid(np.arange(0,1,0.4), np.arange(0,1,0.4), sparse=False, indexing='ij')
print(rhs_fun(func,xv,yv))

