import time
from helper_functions import *
from cgls import *
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D

# Solving the system for the test function 2
m     = 50

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

u            = solve_sys(lap_u_excact_0,X[1:-1,1:-1],Y[1:-1,1:-1],g,m,f_lp5,method='9-point')			# Solution

#Plotting result
fig   = plt.figure(figsize=(15,10))
ax    = fig.gca(projection='3d')
ax.plot_surface(X[1:-1,1:-1],
				Y[1:-1,1:-1],
				u.reshape(m,m), 
				rstride=3, 
				cstride=3, 
				alpha=0.7,
				label='Solved')

ax.plot_surface(X[1:-1,1:-1], 
				Y[1:-1,1:-1], 
				u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1]),
				rstride=3, 
				cstride=3, 
				alpha=0.4,
				color='green',
				label='True')

cset  = ax.contour(X[1:-1,1:-1],
				   Y[1:-1,1:-1], 
				   u.reshape(m,m), 
				   zdir='z', 
				   offset=np.amin(u), 
				   cmap=cm.coolwarm)

cset  = ax.contour(X[1:-1,1:-1], 
				   Y[1:-1,1:-1], 
				   u.reshape(m,m), 
				   zdir='x', 
				   offset=0, 
				   cmap=cm.coolwarm)

cset  = ax.contour(X[1:-1,1:-1], 
				   Y[1:-1,1:-1], 
				   u.reshape(m,m), 
				   zdir='y', 
				   offset=0, 
				   cmap=cm.coolwarm)

#ax.legend()
plt.plot(Y,X,color='blue')
plt.plot(X,Y,color='orange')
ax.set_xlabel('X')
ax.set_xlim(0, 1)
ax.set_ylabel('Y')
ax.set_ylim(0, 1)
ax.set_zlabel('Z')
ax.set_zlim(np.amin(u), np.amax(u))
plt.show()


ss  = np.arange(10,200,10)
sol = np.zeros((len(ss)))

for i in range(len(ss)):
	# Structuring elements
	lp9          = np.array([[1,4,1],[4,-20,4],[1,4,1]]) 													# 9-point Laplace structuring element
	lp5          = np.array([[0,1,0],[1,-4,1],[0,1,0]])  													# 5-point Laplace structuring element
	
	# Domain Discritization
	m            = ss[i]								 													# Number of grid points
	h            = 1/(m+1)				   				 													# Step size
	x            = np.linspace(0, 1, m+2)																	# X range
	y            = np.linspace(0, 1, m+2)				 													# Y range
	X, Y         = np.meshgrid(x, y)					 													# Domain meshgrid
	t            = u_excact_0(X,Y)

	f_lp5		 = (1/12)*(t[0:-2,1:-1]+t[2:,1:-1]+t[1:-1,0:-2]+t[1:-1,2:]-4*t[1:-1,1:-1])	 				# 5-point laplacian of right-hand side

	# Boundary conditions
	g            = t 																						# Full rhs
	g[1:-1,1:-1] = 0																						# Setting interior to zero
	g            = (1/(6*h**2))*signal.convolve2d(g,lp9,mode='valid') 										# Final boundary condition by convulution
	
	u            = solve_sys(lap_u_excact_0,X[1:-1,1:-1],Y[1:-1,1:-1],g,m,f_lp5,method='9-point')		    # Solution
	sol[i]       = np.max(np.abs(u.reshape(m,m)-u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1])))						# Error calculation

h = np.array([1/(x+1) for x in ss])
plt.loglog(h,sol,label='Data')
plt.loglog(h,(h)**2,label='O(h²)')
plt.loglog(h,(h)**4,label='O(h⁴)')
plt.legend(loc='best')
plt.show()
