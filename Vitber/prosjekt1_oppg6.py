# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:24:46 2018

@author: mariadahle
"""

import numpy as np
import sympy as sy
import scipy.special as ss
import pickle
import time
import matplotlib.pyplot as plt
from analytiskF import analytical_solution
from smarte_funksjoner import chebyshev,Lagrange,rho,K,Newton_Cotes,fredholm_lhs,fredholm_rhs


start_t= time.time()
F = pickle.load( open( "F.pkl", "rb" ) )
end_t  = time.time()
print("Initialization took %f s." % (end_t-start_t))

N_eval = 30
a = 0
b = 1
gamma = -2
omega = 3*np.pi
delta = 10**(-3)
d1 = 0.025
xs = chebyshev(a,b,N_eval)
xc = chebyshev(a,b,N_eval)

def finn_b(d):
    F_eval = F(xc,d)
    F_error = np.zeros(len(xc))

    for i in range(len(xc)):
        F_error[i] = F_eval[i]*(1+np.random.uniform(-delta,delta,N_eval)[i])
    return F_eval,F_error

F_eval1, F_error1 = finn_b(0.025)
F_eval2, F_error2 = finn_b(0.25)
F_eval3, F_error3 = finn_b(2.5)

plt.figure()
plt.plot(xc,F_eval1)
plt.plot(xc,F_error1)
plt.plot(xc,F_eval2)
plt.plot(xc,F_error2)
plt.plot(xc,F_eval3)
plt.plot(xc,F_error3)
plt.title('b')
plt.show()

analytisk_rho = rho(omega,gamma,xs,N_eval)

xq_old, w = np.polynomial.legendre.leggauss(N_eval**2)
xq = 0.5*(xq_old + 1)*(b - a) + a
A = fredholm_lhs(xc, xs, xq, w/2)
B = fredholm_rhs(xc, F_eval1)
losning = np.linalg.solve(A,B)

plt.figure()
plt.plot(xs,analytisk_rho)
plt.show()



