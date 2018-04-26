import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt

def eigenvalues_1d_lap(m,omega):
    idx = np.int32(round(m/2))
    yp = np.zeros((m,1))
    h = 1/(m+1)
    for i in range(idx,m):
        yp[i] = (1-omega) + omega*np.cos((i+1)*np.pi*h)
    return np.max(np.abs(yp))

opt =lambda omega: eigenvalues_1d_lap(100,omega)
xopt = scipy.optimize.fmin(func=opt,x0 = 1)
print("1D optimal value for relaxation of Eigeinvalues are: {}".format(xopt[0]))

def eigenvalues_2d_lap(m,omega):
    idx = np.int32(round(m/2))
    ypq = np.zeros((idx,idx))
   # print(ypq.shape)
    h = 1/(m+1)
    for i in range(idx,m):
        for j in range(idx,m):
            ypq[i-idx,j-idx] = (1-omega) +(1/2)*omega*((np.cos((i+1)*np.pi*h)) + (np.cos((j+1)*np.pi*h)))
    return np.max(np.abs(ypq.flatten()))

def eigenvalues_2d_lap_ret(m,omega):
    ypq = np.zeros((m,m))
    h = 1/(m+1)
    for i in range(m):
        for j in range(m):
            ypq[i,j] = (1-omega) + omega*((np.cos((i+1)*np.pi*h)-1) + (np.cos((j+1)*np.pi*h)-1))
    return ypq
            

opt =lambda omega: eigenvalues_2d_lap(100,omega)
xopt = scipy.optimize.fmin(func=opt,x0 = 2)
print("2D optimal value for relaxation of Eigeinvalues are: {}".format(xopt[0]))

m = np.arange(4,20,2)
n = 200
omega = np.linspace(0,2,n)
yp = np.zeros((n,len(m)))
# print(ypq.shape)
h = 1/(m+1)
#omega=1
plt.figure(figsize=(15,10))

for i in range(len(m)):
    for j in range(len(omega)):
        yp[j,i] = eigenvalues_2d_lap(m[i],omega[j])
for i in range(len(m)):
	plt.plot(omega,yp[:,i],label='Grid size {}'.format(m[i]))
plt.legend(loc='best',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("./eigenvalues.pdf")
plt.show()
