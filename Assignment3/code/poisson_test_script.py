import time
from helper_functions import *
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import cg

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
g            = u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1])													# Full rhs
g[1:-1,1:-1] = 0																						# Setting  boundary condition by convulution

# Solving system
u            = solve_sys(u_excact_0,X,Y,g,m,method='9-point')			                                # Solution

T = Amult(lap_u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1]),m)
true = -poisson5(m)*(-lap_u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1])).flatten().T

#plot_pois(X,Y,true,u_excact_0)
#plot_pois(X,Y,T,u_excact_0)
sol = MS_cgls(m,-rhs_fun(u_excact_0,X,Y,g),lap_u_excact_1(X[1:-1,1:-1],Y[1:-1,1:-1])*0,maxIter=10000)
#sol = cg(-poisson5(m).todense(),-u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1]).flatten()+g.flatten())
plot_pois(X,Y,sol[0].reshape(m,m),u_excact_0)
#Plotting result
#plot_pois(X,Y,u,u_excact_1)

ss  = np.arange(10,200,10)
sol = np.zeros((len(ss)))

for i in range(len(ss)):
	# Structuring elements
	lp9          = np.array([[1,4,1],[4,-20,4],[1,4,1]]) 												# 9-point Laplace structuring element
	
	# Domain Discritization
	m            = ss[i]								 												# Number of grid points
	h            = 1/(m+1)				   				 												# Step size
	x            = np.linspace(0, 1, m+2)																# X range
	y            = np.linspace(0, 1, m+2)				 												# Y range
	X, Y         = np.meshgrid(x, y)					 												# Domain meshgrid

	# Boundary conditions
	g            = u_excact_0(X,Y)																		# Full rhs
	g[1:-1,1:-1] = 0																					# Setting interior to zero
	g            = (1/(6*h**2))*signal.convolve2d(g,lp9,mode='valid') 									# Final boundary condition by convulution
	
	u            = solve_sys(u_excact_0,X,Y,g,m,method='9-point')	                             	    # Solving for system
	sol[i]       = np.max(np.abs(u.reshape(m,m)-u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1])))					# Error calculation

# Error plot
h = np.array([1/(x+1) for x in ss])                                                                     # Step size vector
#err_plot(h,sol,s_order=2,e_order=4)                                                                    # Creating the error plot
#plot_pois(X,Y,u,u_excact_0)
