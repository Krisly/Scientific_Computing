import numpy as np
import matplotlib.pyplot as plt

xmin = -30
xmax = 30
ymin = -30
ymax = 30
xv, yv = np.meshgrid(np.arange(xmin,xmax,0.1), np.arange(ymin,ymax,0.1), sparse=False, indexing='ij')

a = xv + 1j*yv
C = np.zeros(a.shape,dtype=np.float64)
A = np.matrix([[(88-7*np.sqrt(6))/360,(296-169*np.sqrt(6))/(1800),(-2+3*np.sqrt(6))/225],
	          [(296+169*np.sqrt(6))/(1800),(88+7*np.sqrt(6))/360,(-2-3*np.sqrt(6))/225],
	          [(16-np.sqrt(6))/36,(16+np.sqrt(6))/36,1/9]],dtype=np.float64)
b = np.array([(16-np.sqrt(6))/36,(16+np.sqrt(6))/36,1/9])

def fill_array(C,A,b,a):
	nrow = A.shape[0]
	ncol = A.shape[1]

	for i in range(len(a)):
		for k in range(len(a)):
			z      = a[i,k]
			I      = np.eye(nrow,ncol)
			e 	   = np.ones((nrow,1))
			eiAinv = np.linalg.inv(I-z*A)
			c_tmp  = 1 + z*b.T*eiAinv*e
			if np.absolute(c_tmp) > 1:
				C[k,i] = 1
			else:
				C[k,i] = np.absolute(c_tmp)
	return C


t = fill_array(C,A,b,a)
#print(t/np.amax(t))
#plt.scatter(a.real,a.imag, c = t,cmap='hsv')
plt.imshow(t,cmap='jet',extent=[xmin, xmax, ymax, ymin],interpolation='bilinear')
plt.colorbar()
plt.show()