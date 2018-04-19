from helper_functions import *
from cgls import *
# Solving the system for the test function 2
m = 200
X,Y,u = solve_sys(lap_test_fun1,0,m,ep=[0,1,0,1],method='9-point')
Nx = Ny = m
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)
sol = cgls(poisson9(m).todense(), np.asmatrix(rhs_fun(lap_test_fun1,X,Y,0)),np.asmatrix(u))

# Plotting the result
fig = plt.figure(figsize=(15,10))
ax = fig.gca(projection='3d')
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

# Plotting the result
u = sol[:,99].reshape(m,m)
fig = plt.figure(figsize=(15,10))
ax = fig.gca(projection='3d')
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
m = np.arange(10,100,10)
sol = np.zeros((len(m)))

for i in range(len(m)):
	X,Y,u = solve_sys(lap_test_fun1,0,m[i],ep=[0,1,0,1],method='9-point')
	sol[i] = np.amax(np.abs(u.reshape(m[i],m[i]) - test_fun1(X,Y)))

h = np.array([1/(x+1) for x in m])
plt.plot(np.log(h),np.log(sol),label='Data')
plt.legend(loc='best')
plt.show()
'''