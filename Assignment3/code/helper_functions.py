import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.sparse.linalg import spsolve
from scipy import signal
from scipy import misc
from numpy.linalg import inv

def cgls( A,b,x0 = 'None',maxIter=1000):
    '''
     The cgls function use an iterative method to solve a Linear system of the form
     Ax=b
    
     INPUT ARGUMENTS
       A       : Is the system matrix 
       b       : Is the vector containing the information of the right-hand side
       maxIter : The number of iterations for the algoritmen to tun through
       xk      : A start guess for the algoritment if not provided the zero vector will be 
                 assumed!!

     OUTPUT
        X : Is a matrix containing a soltion for each iteration hence a matrix
            with as many rows as A and colums as maxIter
     

    # Sets x0 if not provided (as the zero vector)
    if x0 == 'None':
        x0 = np.zeros((A.shape[1],1))
    '''     
    # Check data type of the System-Matrix
    if not isinstance(A,np.matrix):
        print('Wrong Data Type: A must be a matrix')
        raise
    
    # Checks thats the right-hand side is a vectord
    if not isinstance(b,np.matrix):
        error('Wring Data Type: b must be a vector (np.matrix)')
        raise  
    
    # Check the maximum number of iterations
    if maxIter < 1:
        error('Maximum iterations cannot be less than one')
        raise
    
    # Makes sure the right hand side is a column vector 
    if b.shape[0]<b.shape[1]:
       b = b.T

    # Makes sure the initial guess is a column vector 
    if x0.shape[0]<x0.shape[1]:
        x0 = x0.T
    
    C = A.shape[0]               # Gets the number of columns of A
    sol = np.zeros((C,maxIter))  # Preallocates the matrix X
    
    r = b - A*x0
    p = A.T*r
    norm = p.T*p;
    
    for i in  range(maxIter): 
        # Updates the initial guess, x0 and r.
        temp_p = A*p
        ak = (norm/(temp_p.T*temp_p)).item()
        x0 = x0+ak*p
        r = r-ak*temp_p
        p2 = A.T*r;
    
        # Updates the vector d
        nn = p2.T*p2
        beta = (nn/norm).item()
        norm = nn
        p = p2 + beta*p;
        
        # Saves the solution to X
        sol[:,i]=x0.T;
    return sol

def poisson5(m):
	# Calculates the 5-point Laplacian system matrix A
	e = np.repeat(1,m)
	S = sp.sparse.spdiags(np.array([e,-2*e,e]),np.array([-1,0,1]),m,m)
	I = sp.sparse.eye(m)
	A = sp.sparse.kron(I,S) + sp.sparse.kron(S,I)
	A =(m+1)**2*A
	return A

def poisson9(m):
	# Calculates the 9-point Laplacian system matrix A
	e = np.repeat(1,m)
	S = sp.sparse.spdiags(np.array([-e, -10*e ,-e]), np.array([-1, 0, 1]), m, m)
	I = sp.sparse.spdiags(np.array([-1/2*e, e ,-1/2*e]), np.array([-1, 0 ,1]), m, m)
	A = 1/6*(m+1)**2*(sp.sparse.kron(I,S)+sp.sparse.kron(S,I))
	return A

def rhs_fun(f,x,y,g):
	'''
	This calculates the right hand-side used to solved the system Au=f for the poisson equation in a
	2D square domain, it is for the function "solve_sys". 
	'''
	fv    = f(x,y)
	h     = 1/(x.shape[0]+1)
	f_lp5 = (h**2/12)*(fv[0:-2,1:-1]+fv[2:,1:-1]+fv[1:-1,0:-2]+fv[1:-1,2:]-4*fv[1:-1,1:-1])	 			# 5-point laplacian of right-hand side
	rhs   = (fv[1:-1,1:-1]+f_lp5) - g 
	return rhs.flatten()

def solve_sys(f,x,y,g,m,method='5-point'):
	'''
	This function solves the a 2D problem with a square domain [a ; b] x [c ; d], using
	either the 5-point Laplacian or 9-point Laplacian, with specified parameters:

	fun    : The function which is being solved

	u      : The bundary conditions

	m      : The amount of grid-points. The realtioship of m to the step-size is h = 1/(m+1)

	ep     : This is the endpoints of the 2D-domain

	method : This specify whether to use the 5-point or 9-point Laplacian
	'''

	if method == '5-point':
		return spsolve(poisson5(m), rhs_fun(f,y,y,g))
	elif method == '9-point':
#		return inv(poisson9(m).todense())*np.asmatrix(rhs_fun(f,x,y,g)).T
		return spsolve(poisson9(m), rhs_fun(f,x,y,g))

def u_excact_0(x,y):
	# Test function used to evaluate convergence rate

	return np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)

def lap_u_excact_0(x,y):
	# Test function used to evaluate convergence rate

	return -16*np.pi**2*(2*np.sin(4*np.pi*(x+y))+np.cos(4*np.pi*x*y)*(x**2+y**2))

def u_excact_1(x,y):
	# Test function used to evaluate convergence rate

	return x**2+y**2

def lap_u_excact_1(x,y):
    # Returns the laplacian of test function excat_1
    
	return 4*np.ones(x.shape)
	
def u_excact_2(x,y):
	# Test function used to evaluate convergence rate

	return np.sin(2*np.pi*np.abs(x-y)**(2.5))

def Amult(U,m):    
    # Matrix less matrix product
    U = U.reshape(m,m)
    h = 1/(m+1)
    se = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    U = signal.convolve2d(U,se,mode='same')
    return ((-1/h**2)*U).flatten()
    
def UR():
    # Underer relaxion of the Jacobian
    
    return 0
    
def plot_pois(X,Y, u,f):
    # Poisson plot function
    
    m = X.shape[0]-2
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
				f(X[1:-1,1:-1],Y[1:-1,1:-1]),
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
    
def err_plot(h,sol,s_order=0,e_order=4):
    plt.loglog(h,sol,label='Data')
    
    for i in range(s_order,e_order,1):
        plt.loglog(h,(h)**(i),label=r'$O(h^{})$'.format(i))
    
    plt.legend(loc='best')
    plt.show()
    
    
    
def MS_cgls(m,b,x0 = 'None',maxIter=1000):
    '''
     The cgls function use an iterative method to solve a Linear system of the form
     Ax=b
    
     INPUT ARGUMENTS
       A       : Is the system matrix 
       b       : Is the vector containing the information of the right-hand side
       maxIter : The number of iterations for the algoritmen to tun through
       xk      : A start guess for the algoritment if not provided the zero vector will be 
                 assumed!!

     OUTPUT
        X : Is a matrix containing a soltion for each iteration hence a matrix
            with as many rows as A and colums as maxIter
     

    # Sets x0 if not provided (as the zero vector)
    if x0 == 'None':
        x0 = np.zeros((A.shape[1],1))
    '''   
    x0 = np.asmatrix(x0.flatten()).T
    sol = np.zeros((m**2,maxIter))  # Preallocates the matrix X
    
    r = np.asmatrix(b - Amult(x0,m)).T
    p = np.asmatrix(Amult(r,m)).T
    norm = p.T*p;
    for i in  range(maxIter): 
        # Updates the initial guess, x0 and r.
        temp_p = np.asmatrix(Amult(p,m)).T
        ak = (norm/(temp_p.T*temp_p)).item()
        x0 = x0+ak*p
        r = r-ak*temp_p
        p2 = np.asmatrix(Amult(r,m)).T;
    
        # Updates the vector d
        nn = p2.T*p2
        beta = (nn/norm).item()
        norm = nn
        p = p2 + beta*p;
        
    # Returns the solution to X
    return x0.T

def smooth(U,omega,m,F):
    return (1-omega)*U + omega*(1 + U)

print(poisson5(3).todense())
