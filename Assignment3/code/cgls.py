import numpy as np

def cgls( A,b,x0 = 'None',maxIter=100):
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
    ''' 

    # Sets x0 if not provided (as the zero vector)
    if x0 == 'None':
        x0 = np.zeros((A.shape[1],1))
     
    # Check data type of the System-Matrix
    if not isinstance(A,np.matrix):
        print('Wrong Data Type: A must be a matrix')
        raise
    
    # Checks thats the right-hand side is a vector
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
