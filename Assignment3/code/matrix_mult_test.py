import numpy as np
from helper_functions import *
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


m= 100
h = 1/(m+1)
b = np.ones((m**2,1))
A = -poisson5(m).todense()
x = np.arange(0,1-h,h)
print(len(x))
y = np.flip(x,axis=0)
print(x,y)
X, Y = np.meshgrid(x, y)
b = np.sin(X*Y)
res1 = A*np.asmatrix(b.flatten()).T
res2 = Amult(b,m)
fig   = plt.figure(figsize=(15,10))
ax    = fig.gca(projection='3d')
ax.plot_surface(X[1:-1,1:-1],
				Y[1:-1,1:-1],
				res1.reshape(m,m)[1:-1,1:-1]-res2.reshape(m,m)[1:-1,1:-1], 
				rstride=3, 
				cstride=3, 
				alpha=1.0,
				label='Solved',color='red')
for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(40) 

for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(40) 
for tick in ax.zaxis.get_major_ticks():
                tick.label.set_fontsize(40) 
plt.savefig('./mf_mult_res.pdf')

plt.show()


