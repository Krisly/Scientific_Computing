import time
from helper_functions import *
from cgls import *
from scipy import signal
# Solving the system for the test function 2
m     = 100
'''
start = time.time()
X,Y,u = solve_sys(test_fun,0,m,ep=[0,1,0,1],method='9-point')
end   = time.time()
print('Elapsed time: {}'.format(end-start))

Nx = Ny = m
x       = np.linspace(0, 1, Nx)
y       = np.linspace(0, 1, Ny)
X, Y    = np.meshgrid(x, y)
start   = time.time()
sol     = cgls(poisson9(m).todense(), np.asmatrix(rhs_fun(lap_test_fun1,X,Y,0)),0.2*np.asmatrix(u))
end     = time.time()
print('Elapsed time: {}'.format(end-start))
# Plotting the result
fig    = plt.figure(figsize=(15,10))
ax     = fig.gca(projection='3d')
ax.plot_surface(X, Y, u.reshape(m,m), rstride=8, cstride=8, alpha=0.7)
cset   = ax.contour(X, Y, u.reshape(m,m), zdir='z', offset=np.amin(u), cmap=cm.coolwarm)
cset   = ax.contour(X, Y, u.reshape(m,m), zdir='x', offset=0, cmap=cm.coolwarm)
cset   = ax.contour(X, Y, u.reshape(m,m), zdir='y', offset=0, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(0, 1)
ax.set_ylabel('Y')
ax.set_ylim(0, 1)
ax.set_zlabel('Z')
ax.set_zlim(np.amin(u), np.amax(u))
plt.show()


# Plotting the result
u    = sol[:,99].reshape(m,m)
fig  = plt.figure(figsize=(15,10))
ax   = fig.gca(projection='3d')
ax.plot_surface(X, Y, u.reshape(m,m), rstride=8, cstride=8, alpha=0.7)
cset = ax.contour(X, Y, u.reshape(m,m), zdir='z', offset=np.amin(u), cmap=cm.coolwarm)
cset = ax.contour(X, Y, u.reshape(m,m), zdir='x', offset=0, cmap=cm.coolwarm)
cset = ax.contour(X, Y, u.reshape(m,m), zdir='y', offset=0, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(0, 1)
ax.set_ylabel('Y')
ax.set_ylim(0, 1)
ax.set_zlabel('Z')
ax.set_zlim(np.amin(u), np.amax(u))
plt.show()



'''
# Plotting the result
x = np.linspace(0, 1, m+2)
y = np.linspace(0, 1, m+2)
X, Y = np.meshgrid(x, y)
g = u_excact_0(X,Y)
g[1:-1,1:-1] =0
he = np.array([[1,4,1],[4,-20,4],[1,4,1]])
g=signal.convolve2d(g,he,mode='valid')

u = solve_sys(lap_u_excact_0,X[1:-1,1:-1],Y[1:-1,1:-1],g,m,method='9-point')
print(u.shape)
#u     = cgls(poisson9(m).todense(), np.asmatrix(rhs_fun(test_fun,X,Y,0)),np.asmatrix(test_fun(X,Y).flatten()))
#u     = u.reshape(m,m) - test_fun(X,Y)
fig   = plt.figure(figsize=(15,10))
ax    = fig.gca(projection='3d')
ax.plot_surface(X[1:-1,1:-1], Y[1:-1,1:-1], u.reshape(m,m), rstride=8, cstride=8, alpha=0.7,label='Solved')
ax.plot_surface(X[1:-1,1:-1], Y[1:-1,1:-1], u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1]), rstride=8, cstride=8, alpha=0.4,color='green',label='True')
cset  = ax.contour(X[1:-1,1:-1], Y[1:-1,1:-1], u.reshape(m,m), zdir='z', offset=np.amin(u), cmap=cm.coolwarm)
cset  = ax.contour(X[1:-1,1:-1], Y[1:-1,1:-1], u.reshape(m,m), zdir='x', offset=0, cmap=cm.coolwarm)
cset  = ax.contour(X[1:-1,1:-1], Y[1:-1,1:-1], u.reshape(m,m), zdir='y', offset=0, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(0, 1)
ax.set_ylabel('Y')
ax.set_ylim(0, 1)
ax.set_zlabel('Z')
ax.set_zlim(np.amin(u), np.amax(u))
plt.show()


m   = np.arange(10,200,10)
sol = np.zeros((len(m)))

for i in range(len(m)):
	x            = np.linspace(0, 1, m[i]+2)
	y            = np.linspace(0, 1, m[i]+2)
	X, Y         = np.meshgrid(x, y)
	g            = u_excact_0(X,Y)
	g[1:-1,1:-1] = 0
	he           = np.array([[1,4,1],[4,-20,4],[1,4,1]])
	g            = signal.convolve2d(g,he,mode='valid')

	u            = solve_sys(lap_u_excact_0,X[1:-1,1:-1],Y[1:-1,1:-1],g,m[i],method='9-point')
	sol[i]       = np.max(np.abs(u.reshape(m[i],m[i])-u_excact_0(X[1:-1,1:-1],Y[1:-1,1:-1])))

h = np.array([1/(x+1) for x in m])
plt.loglog(h,sol,label='Data')
#plt.loglog(h,h,label='O(h)')
plt.loglog(h,(h)**2,label='O(h²)')
#plt.loglog(h,h**3,label='O(h³)')
#plt.loglog(h,h**4,label='O(h⁴)')
plt.legend(loc='best')
plt.show()


'''
plt.figure()
plt.plot(X,Y,color='teal')
plt.plot(Y,X,color='orange')
ax = plt.gca()
plt.show()
'''