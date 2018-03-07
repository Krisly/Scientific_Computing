"""
Created on Mon Feb 26 11:29:32 2018

@author: root
"""
from numsolver import numerical_solvers
import numpy as np
import matplotlib.pyplot as plt

# Sets Plotting parameters
font_size = 15
plt.rcParams['figure.figsize'] = (15,7)
plt.rc('font',   size=font_size)       # controls default text sizes
plt.rc('axes',  titlesize=font_size)   # fontsize of the axes title
plt.rc('axes',   labelsize=font_size)  # fontsize of the x any y labels
plt.rc('xtick',  labelsize=font_size)  # fontsize of the tick labels
plt.rc('ytick',  labelsize=font_size)  # fontsize of the tick labels
plt.rc('legend', fontsize=font_size)   # legend fontsize
plt.rc('figure', titlesize=font_size)  # # size of the figure title

# Setting up solver parameters
mu              = 3
x0              = np.array([2.0,0.0])
tend            = 20
numsolv1        = numerical_solvers()
numsolv1.param  = mu
numsolv1.x0     = x0
numsolv1.maxit  = 100
numsolv1.t      = np.array([0,tend])
numsolv1.dt     = 0.01
t               = np.arange(0,tend+0.01,0.01)
numsolv1.absTol = 10**(-8)
numsolv1.relTol = 10**(-3)
numsolv1.epstol = 0.8
absTol_as       = 10**(-4) 


mu = 3
sigma = 0.5
x0 = np.array([0.5, 0.5])
p = np.array([mu, sigma])

tf = 5*mu
nw = 1
N = 5000
Ns = 5
seed = 100

W,T,dW=numsolv1.ScalarStdWienerProcess(tf,N,nw,Ns);

