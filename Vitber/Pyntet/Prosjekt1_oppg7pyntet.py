# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:21:48 2018

@author: helen
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
#from analytisk_F import analytical_solution
from prosjekt1_formler import chebyshev, fredholm_lhs, rho


F = pickle.load( open( "F.pkl", "rb" ) )


Nc = Ns = 30
a = 0
b = 1
gamma = -2
omega = 3*np.pi
delta = 10**(-3)
d1 = 0.25
d2 = 2.5
xs = chebyshev(a,b,Ns)
xc = chebyshev(a,b,Nc)
F_eval = F(xc,d1)

xq_old, w = np.polynomial.legendre.leggauss(Nc**2)
xq = 0.5*(xq_old + 1)*(b - a) + a

def b_pert(d):
    F_eval = F(xc,d)
    F_error = np.zeros(Nc)

    for i in range(Nc):
        F_error[i] = F_eval[i]*(1+np.random.uniform(-delta,delta,Nc)[i])
    return F_error
    
rho_analytical = rho(omega,gamma,xc,Nc)

lambda_list = np.logspace(-14,1,Nc)
#lambda_list = np.geomspace(10**(-14),10,N_eval)

def error(d,lambda_list):
    A = fredholm_lhs(xc, xs, xq, w/2, d)
    A_t = np.transpose(A)
    errorlist = np.zeros(len(lambda_list))
    for i in range(len(lambda_list)):
        left_side = np.add(np.matmul(A_t,A),np.multiply(lambda_list[i],np.eye(Nc)))
        right_side = np.dot(A_t,b_pert(d))
        rho_numerical = np.linalg.solve(left_side,right_side)
        single_error_list = np.zeros(Nc)
        for j in range(len(rho_numerical)):
            single_error_list[j] = abs(rho_numerical[j] - rho_analytical[j])
        errorlist[i] = np.linalg.norm(single_error_list,np.inf)
    return errorlist

plt.figure()
plt.loglog(lambda_list,error(d1,lambda_list))
plt.xlabel('Error as a function of $\lambda$, with depth $d=0.25$')
plt.ylabel('$Error(\lambda)$')
plt.show()

plt.figure()
plt.loglog(lambda_list,error(d2,lambda_list))
plt.xlabel('Error as a function of $\lambda$, with depth $d=2.5$')
plt.ylabel('$Error(\lambda)$')
plt.show()

lamb1 = 10**(-4)
lamb2 = 10**(-10)

def rho_numerical(d,lamb):
    A = fredholm_lhs(xc, xs, xq, w/2, d)
    A_t = np.transpose(A)
    left_side = np.add(np.matmul(A_t,A),np.multiply(lamb,np.eye(Nc)))
    right_side = np.dot(A_t,b_pert(d))
    rho_numerical = np.linalg.solve(left_side,right_side)
    return rho_numerical

plt.figure()
plt.plot(xc,rho_numerical(d1,lamb1))
plt.xlabel(r'Numerical $\rho$ with $\lambda=10^{-4}$ and $d=0.25$')
plt.ylabel(r'$\rho(x_c)$')
plt.show()

plt.figure()
plt.plot(xc,rho_numerical(d2,lamb2))
plt.xlabel(r'Numerical $\rho$ with $\lambda=10^{-10}$ and $d=2.5$')
plt.ylabel(r'$\rho(x_c)$')
plt.show()
