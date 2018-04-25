import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from helper_functions import poisson9
from numpy.linalg import inv
from scipy import signal

u0e = lambda x,y: np.sin(4*np.pi*(x+y)) + np.cos(4*np.pi*x*y)
lu0e = lambda x,y: -16*np.pi**2*(2*np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)*(x**2+y**2))

m = 50
h = 1/(m+1)
x = np.linspace(0,1,m+2)
y = np.linspace(0,1,m+2)
X,Y = np.meshgrid(x,y)
X_int = X[1:-1,1:-1]
Y_int = Y[1:-1,1:-1]

se = np.array([[1,4,1],[4,-20,4],[1,4,1]])

g = u0e(X,Y)
g[1:-1,1:-1] = 0
g = signal.convolve2d(g,se,mode='valid')

A = poisson9(m)
u = lu0e(X_int,Y_int)-((h**2)/12)*g

sol = inv(A.todense())*np.asmatrix(u.flatten()).T

fig   = plt.figure(figsize=(15,10))
ax    = fig.gca(projection='3d')
ax.plot_surface(X_int,
				Y_int,
				sol.reshape(m,m), 
				rstride=3, 
				cstride=3, 
				alpha=1.0,
				label='Solved')

ax.plot_surface(X_int,
				Y_int,
				u0e(X_int,Y_int), 
				rstride=3, 
				cstride=3, 
				alpha=1.0,
				label='Solved')

plt.show()
