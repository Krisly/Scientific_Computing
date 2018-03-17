import numpy as np
import matplotlib.pyplot as plt

xmin = -30
xmax = 30
ymin = -30
ymax = 30
xv, yv = np.meshgrid(np.arange(xmin,xmax,0.05),
                     np.arange(ymin,ymax,0.05),
                     sparse=False, indexing='ij')

a = xv + 1j*yv
C = np.zeros(a.shape,dtype=np.float64)
A = np.matrix([[(88-7*np.sqrt(6))/360,(296-169*np.sqrt(6))/(1800),(-2+3*np.sqrt(6))/225],
	          [(296+169*np.sqrt(6))/(1800),(88+7*np.sqrt(6))/360,(-2-3*np.sqrt(6))/225],
	          [(16-np.sqrt(6))/36,(16+np.sqrt(6))/36,1/9]],dtype=np.float64)
b = np.array([(16-np.sqrt(6))/36,(16+np.sqrt(6))/36,1/9])
#A = np.matrix([[0,0,0,0],
#	          [1/2,0,0,0],
#	          [0,1/2,0,0],
#	          [0,0,1,0]])
#b = np.array([1/6,1/3,1/3,1/6])

#A = np.matrix([[0,0,0,0,0,0,0],
#	          [1/5,0,0,0,0,0,0],
#	          [3/40,9/40,0,0,0,0,0],
#	          [44/45,-56/15,32/9,0,0,0,0],
#	          [19372/6561,-25360/2187,64448/6561,-212/729,0,0,0],
#	          [9017/3168,-355/33,46732/5247,49/176,-5103/18656,0,0],
#	          [35/384,0,500/1113,125/192,-2187/6784,11/84,0]])
#print(A,A.shape,type(A))
#b = np.array([35/384,0,500/1113,125/192,-2187/6784,11/84,0])
#b = np.array([5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40])


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
#t[t<1]=0
#print(t)
#print(t/np.amax(t))
#plt.scatter(a.real,a.imag, c = t,cmap='hsv')
plt.imshow(t,cmap='jet',extent=[xmin, xmax, ymax, ymin],interpolation='bilinear')
plt.colorbar()
plt.plot((xmin, xmax), (0, 0), '--k')
plt.plot((0, 0), (ymin, ymax), '--k')
plt.savefig('./figs/abs_stab_radau_col.pdf')
plt.show()
