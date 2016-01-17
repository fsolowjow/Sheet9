from __future__ import division
import numpy as np
import scipy
from scipy import special
from scipy import linalg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from matplotlib import animation

def coeff_a(n , alpha , beta):
	return (2/(2*n + alpha + beta ) * np.sqrt((n*(n+alpha+beta)*(n+alpha)*(n+beta)) / ((2*n+alpha+beta-1)*(2*n+alpha+beta+1))))

def coeff_b(n , alpha , beta):
	return -(alpha*alpha - beta*beta)/((2*n+alpha+beta)*(2*n+alpha+beta+2))

#a)	
def jacobiP(x, alpha, beta, n):
	if(n==0):	return np.full ( np.size(x) , np.sqrt( np.power(2., -alpha - beta - 1) * scipy.special.gamma(alpha+beta+2) / (scipy.special.gamma(alpha+1)* scipy.special.gamma(beta+1))))
	if(n==1):	return 0.5 * np.sqrt((alpha+beta+3)/((alpha+1)*(beta+1))) *((alpha+beta+2)*x + alpha - beta) * jacobiP(x,alpha,beta,0)
	if(n>1):	return ( (x*jacobiP(x,alpha,beta, n-1) - coeff_a(n-1,alpha,beta) * jacobiP(x,alpha,beta, n-2) - coeff_b(n-1,alpha,beta) * jacobiP(x,alpha,beta, n-1)) / coeff_a(n,alpha,beta))

#b)	
def jacobiDer(x, alpha, beta, n):
	return np.sqrt(n*(n+1+alpha+beta))*jacobiP(x, alpha+1 , beta+1 , n-1)

#c) b_0 = alpha = beta = 0 => Problem mit unserer definition!
def iniT ( n, alpha , beta):
	T=np.zeros((n+1,n+1))
	for i in range(n+1):
		T[i,i] = coeff_b(i , alpha , beta)
		if(i < n): T[i,i+1] = coeff_a( i+1 , alpha , beta)
		if(i > 0): T[i,i-1] = coeff_a( i , alpha , beta)
	eigenvalue , eigenvector = LA.eig(T)
	
alpha=0
beta=0
n=5
x=np.arange(0.,1.1,0.1)	
	
iniT(6,1,2)