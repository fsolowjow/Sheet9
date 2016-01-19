from __future__ import division
import numpy as np
import scipy
from scipy import special
from scipy import linalg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from matplotlib import animation
from scipy.integrate import quad
from scipy.optimize import brentq

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

#c) b_0 = alpha = beta = 0 => Problem with our definition!
def integrand(x):
	return np.power((1.-x),alpha) * np.power((1+x),beta)

def JacobiQuad( n, alpha , beta):
	T=np.zeros((n+1,n+1))
	for i in range(n+1):
		T[i,i] = coeff_b(i+1 , alpha , beta)	#Since b_0 does not make sense
		if(i < n): T[i,i+1] = coeff_a( i+1 , alpha , beta)
		if(i > 0): T[i,i-1] = coeff_a( i , alpha , beta)
	#quadrature points  = eigenvalues
	#quadrature weights = eigenvector_j[1]^2 * integral w(x), with w(x) = (1-x)^alpha (1+x)^beta. Also wrong defined in lecture notes.
	eigenvalue , eigenvector = LA.eig(T)
	
	quadrature_points = eigenvalue
	weight_integral = quad( integrand , -1 , 1 )[0]
	quadrature_weights = eigenvector[0,:] * weight_integral
	return quadrature_points, quadrature_weights

#d)
#probably typo in constants
def helperfunc(x,alpha,beta,n):
	return (1-x*x) * jacobiDer(x,alpha,beta,n)
	
def JacobiLGL(alpha, beta, n):
	eps=0.000001
	#The quadrature points are the roots of (1-x^2) P'_n = helperfunc
	quadrature_points = np.zeros(n+1)
	quadrature_points[0]=-1
	for i in range(n):
		#calculates the roots of (1-x^2) P'_n
		quadrature_points[i+1] = brentq(helperfunc, quadrature_points[i] + eps , 1 , args=(alpha,beta,n))
	
	quadrature_weights = np.zeros(n+1)
	for i in range(1,n):
		quadrature_weights[i] = 2/(n*(n-1)*jacobiP( quadrature_points[i] , alpha, beta, n))
	quadrature_weights[n]= 2/(n*(n-1))
	quadrature_weights[0]= 2/(n*(n-1))
	return quadrature_points, quadrature_weights	

#e)
def JacobiVMat(x,n):
	points,temp = JacobiLGL(0,0,n)
	V=np.zeros((n+1,n+1))
	for j in range(n+1):
		V[:,j] = jacobiP(points, 0, 0, j)

def JacobiDMat(x,n):
	points,temp = JacobiLGL(0,0,n)
	V=np.zeros((n+1,n+1))
	for j in range(n+1):
		if(j > 0):	V[:,j] = jacobiDer(points, 0, 0, j)		#for j = 0 problem!
				
		
alpha=1
beta=1
n=5
x=np.arange(-1.,1.1,0.1)	
	
a,b=JacobiQuad(n,alpha,beta)
c,d=JacobiLGL(alpha,beta,n)

JacobiVMat(x,n)
JacobiDMat(x,n)
