import time
from helper_functions import *
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D

# Solving the system for the test function 2
m     = 100

# Structuring elements
lp9          = np.array([[1,4,1],[4,-20,4],[1,4,1]]) 													# 9-point Laplace structuring element
lp5          = np.array([[0,1,0],[1,-4,1],[0,1,0]])  													# 5-point Laplace structuring element
#print(lp5)
# Domain Discritization
h            = 1/(m+1)				   				 													# Step size
x            = np.linspace(0, 1, m+2)																	# X range
y            = np.linspace(0, 1, m+2)				 													# Y range
X, Y         = np.meshgrid(x, y)					 													# Domain meshgrid
f_lp5        = (h**2/12)*signal.convolve2d(u_excact_0(Y,X),lp5,mode='valid')					    	# 5-point laplacian of right-hand side
#print(f_lp5)
#print("")

# Boundary conditions
g            = u_excact_0(X,Y)																			# Full rhs
#print(g[1:-1,1:-1])
g[1:-1,1:-1] = 0																						# Setting interior to zero
g            = (1/(6*h**2))*signal.convolve2d(g,lp9,mode='valid') 										# Final boundary condition by convulution

u            = solve_sys(lap_u_excact_0,X,Y,g,m,method='9-point')			                            # Solution

#Plotting result
plot_pois(X,Y,u,u_excact_0)

ss  = np.arange(10,200,10)
sol = np.zeros((len(ss)))

for i in range(len(ss)):
	# Structuring elements
	lp9          = np.array([[1,4,1],[4,-20,4],[1,4,1]]) 												# 9-point Laplace structuring element
	lp5          = np.array([[0,1,0],[1,-4,1],[0,1,0]])  												# 5-point Laplace structuring element
	
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
	
	u            = solve_sys(u_excact_0,X,Y,g,m,method='9-point')	                             	    # Solution
	sol[i]       = np.max(np.abs(u.reshape(m,m)-u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1])))					# Error calculation

h = np.array([1/(x+1) for x in ss])
err_plot(h,sol,s_order=2,e_order=4)

#plot_pois(X,Y,u,u_excact_0)
