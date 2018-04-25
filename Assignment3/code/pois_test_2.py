import time
from helper_functions import *
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import cg

m= 50

# Domain Discritization
h            = 1/(m+1)				   				 													# Step size
x            = np.linspace(0, 1, m+2)																	# X range
y            = np.linspace(0, 1, m+2)				 													# Y range
X, Y         = np.meshgrid(x, y)					 													# Domain meshgrid

sol_o = MS_cgls(m,-rhs_fun(u_excact_0,X,Y,0),lap_u_excact_1(X[1:-1,1:-1],Y[1:-1,1:-1])*0,maxIter=10000)
#sol_s = cg(poisson5(m).todense(),u_excact_1(X[1:-1,1:-1],Y[1:-1,1:-1]).flatten()-g.flatten())

fig   = plt.figure(figsize=(15,10))
ax    = fig.gca(projection='3d')
ax.plot_surface(X[1:-1,1:-1],
				Y[1:-1,1:-1],
				sol_o.reshape(m,m), 
				rstride=3, 
				cstride=3, 
				alpha=1.0,
				label='Solved')

plt.show()
