import time
from helper_functions import *
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D

# Solving the system for the test function 2
m     = 100

# Structuring elements
lp9          = np.array([[1,4,1],[4,-20,4],[1,4,1]]) 													# 9-point Laplace structuring element

# Domain Discritization
h            = 1/(m+1)				   				 													# Step size
x            = np.linspace(0, 1, m+2)																	# X range
y            = np.linspace(0, 1, m+2)				 													# Y range
X, Y         = np.meshgrid(x, y)					 													# Domain meshgrid

# Boundary conditions
g            = u_excact_1(X,Y)																			# Full rhs
g[1:-1,1:-1] = 0																						# Setting interior to zero
g            = (1/(6*h**2))*signal.convolve2d(g,lp9,mode='valid') 										# Final boundary condition by convulution

# Solving system
u            = solve_sys(u_excact_1,X,Y,g,m,method='9-point')			                            # Solution

#Plotting result
plot_pois(X,Y,u,u_excact_1)

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

T = Amult(u,m)
fig   = plt.figure(figsize=(15,10))
ax    = fig.gca(projection='3d')

ax.plot_surface(X[1:-1,1:-1],
				Y[1:-1,1:-1],
				T.reshape(m,m), 
				rstride=3, 
				cstride=3, 
				alpha=0.7,
				label='Solved')

plt.show()
# Error plot
h = np.array([1/(x+1) for x in ss])                                                                     # Step size vector
err_plot(h,sol,s_order=2,e_order=4)                                                                     # Creating the error plot
plot_pois(X,Y,u,u_excact_0)
