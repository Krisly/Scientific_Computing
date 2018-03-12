import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import cProfile


num_methods = {'Classic':
                pd.DataFrame(np.array([[0,1/2,1/2,1],
                                      [1/6,1/3,1/3,1/6],
                                      [0,1/2,0,0],
                                      [0,0,1/2,0],
                                      [0,0,0,1],
                                      [0,0,0,0]]).T,
                columns=['c', 'x','coef0', 'coef1', 'coef2', 'coef3']),
                '3/8-rule':
                pd.DataFrame(np.array([[0,1/3,2/3,1],
                                      [1/8,3/8,3/8,1/8],
                                      [0,1/3,-1/3,1],
                                      [0,0,1,-1],
                                      [0,0,0,1],
                                      [0,0,0,0]]).T,
                columns=['c', 'x','coef0', 'coef1', 'coef2', 'coef3']),
                'Dormand-Prince':
                pd.DataFrame(np.array([[0,1/5,3/10,4/5,8/9,1,1],
                 [35/384,0,500/1113,125/192,-2187/6784,11/84,0],
                 [5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40],
                 [0,1/5,3/40,44/45,19372/6561,9017/3168,35/384],
                 [0,0,9/40,-56/15,-25360/2187,-355/33,0],
                 [0,0,0,32/9,64448/6561,46732/5247,500/1113],
                 [0,0,0,0,-212/729,49/176,125/192],
                 [0,0,0,0,0,-5103/18656,-2187/6784],
                 [0,0,0,0,0,0,11/84],
                 [0,0,0,0,0,0,0]],dtype=np.float64).T,
                columns=['c', 'x','xh','coef0', 'coef1', 'coef2', 'coef3',
                         'coef4','coef5','coef6']),
                'Bogacki–Shampine':
                pd.DataFrame(np.array([[0,1/2,3/4,1],
                 [2/9,1/3,4/9,0],
                 [2/9,1/3,4/9,1/8],
                 [0,1/2,0,2/9],
                 [0,0,3/4,1/3],
                 [0,0,0,4/9],
                 [0,0,0,0]]).T,
                columns=['c', 'x','xh','coef0', 'coef1', 'coef2', 'coef3'])
    }

print(num_methods['Dormand-Prince'])