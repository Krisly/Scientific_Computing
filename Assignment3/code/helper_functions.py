import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.sparse.linalg import spsolve

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
	return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)*(x**y)

def func2(x,y):
	return np.exp(x) - np.exp(y)

m = 100
Nx = Ny = m
x = np.linspace(-5, 5, Nx)
y = np.linspace(-5, 5, Ny)
X, Y = np.meshgrid(x, y)
u = spsolve(poisson5(m), func2(X,Y).flatten().T)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, u.reshape(m,m), rstride=10, cstride=10)

plt.show()

plt.figure()
plt.clf()
plt.plot_surface(X,Y,u.reshape(m,m))
plt.title('X grid $(X_{i,j})_{i,j}$')
plt.xlabel(r'$j$')
plt.ylabel(r'$i$')
plt.colorbar()
plt.show()
#
#plt.figure()
#plt.clf()
#plt.imshow(poisson5(2).todense())
#plt.title('Y grid $(Y_{i,j})_{i,j}$')
#plt.xlabel(r'$j$')
#plt.ylabel(r'$i$')
#plt.colorbar()
#plt.show()