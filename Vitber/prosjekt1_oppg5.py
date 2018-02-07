# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:20:39 2018

@author: mariadahle
"""

import numpy as np
import sympy as sy
import time
import pickle
import matplotlib.pyplot as plt
from analytiskF import analytical_solution
from smarte_funksjoner import chebyshev,Lagrange,rho,K,Newton_Cotes,fredholm_lhs,fredholm_rhs

a = 0
b = 1
Ns = 5
Nc = 5
Nq = 25
d = 0.025
omega = 3*np.pi
gamma = -2
N_eval = np.arange(5,30,1)

xq_old, w = np.polynomial.legendre.leggauss(25)
xq = 0.5*(xq_old + 1)*(b - a) + a
xs = chebyshev(a,b,Ns)
xc = chebyshev(a,b,Nc)


A = fredholm_lhs(xc, xs, xq ,w/2)


start_t= time.time()
F = pickle.load( open( "F.pkl", "rb" ) )
end_t  = time.time()
print("Initialization took %f s." % (end_t-start_t))
#F_eval = F(xc,d)
#b = fredholm_rhs(xc, F_eval)

"""
print(F_eval)
print(b)
print(A)

losning = np.linalg.solve(A,b)
print(losning)

print(rho(omega,gamma,xs,Ns))
"""


def rho_error(Ns,Nc,F,d):
    error = np.zeros(len(Ns))
    for i in range(len(Ns)):
        xs = chebyshev(a,b,Ns[i])
        xc = chebyshev(a,b,Nc[i])
        xq_old, w = np.polynomial.legendre.leggauss(Ns[i]**2)
        xq = 0.5*(xq_old + 1)*(b - a) + a
        A = fredholm_lhs(xc, xs, xq, w/2)
        
        F_eval = F(xc,d)
        B = fredholm_rhs(xc, F_eval)
        
        losning = np.linalg.solve(A,B)
        ny_liste = np.zeros(Ns[i])
        for j in range(Ns[i]):
            ny_liste[j] = rho(omega,gamma,xs,Ns[i])[j]-losning[j]
        error[i] = np.linalg.norm(ny_liste,np.inf)
    return error

Ns_test = np.arange(5,30,1)

#error = rho_error(Ns_test,Ns_test,F,0.025)

error_d1= rho_error(Ns_test,Ns_test,F,0.025)

plt.figure()
plt.plot(Ns_test,error_d1)
plt.plot(Ns_test,rho_error(Ns_test,Ns_test,F,0.03))
plt.plot(Ns_test,rho_error(Ns_test,Ns_test,F,2.5))
plt.xlabel('Ns')
plt.title('Error i rho')
plt.show()


plt.figure()
plt.title('Logaritmisk skalert error i rho')
plt.semilogy(Ns_test,error_d1)
plt.show()




