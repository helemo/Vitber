# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:21:48 2018

@author: helen
"""


import numpy as np
import sympy as sy
import scipy.special as ss
import pickle
import time
import matplotlib.pyplot as plt
from analytisk_F import analytical_solution
from prosjekt1_formler import chebyshev, fredholm_lhs, rho , K , Lagrange, fredholm_rhs, Newton_Cotes

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
d1 = 0.25
d2 = 2.5
xs = chebyshev(a,b,N_eval)
xc = chebyshev(a,b,N_eval)
F_eval = F(xc,d1)

xq_old, w = np.polynomial.legendre.leggauss(N_eval**2)
xq = 0.5*(xq_old + 1)*(b - a) + a
A = fredholm_lhs(xc, xs, xq, w/2)
B = fredholm_rhs(xc, F_eval)

def finn_b_perturbert(d):
    F_eval = F(xc,d)
    F_error = np.zeros(len(xc))

    for i in range(len(xc)):
        F_error[i] = F_eval[i]*(1+np.random.uniform(-delta,delta,N_eval)[i])
    return F_error
    
rho_analytisk = rho(omega,gamma,xc,N_eval)

lambda_liste = np.logspace(-14,1,N_eval)

def error(d,lambda_liste):
    A = fredholm_lhs(xc, xs, xq, w/2)
    A_t = np.transpose(A)
    errorliste = np.zeros(len(lambda_liste))
    #lambda_liste = np.random.uniform(10**(-14),10,30)
    #print(type[A],type[A_t], type[])
    for i in range(len(lambda_liste)):
        venstre_side = np.add(np.matmul(A_t,A),np.multiply(lambda_liste[i],np.eye(N_eval)))
        hoyre_side = np.dot(A_t,finn_b_perturbert(d))
        losning = np.linalg.solve(venstre_side,hoyre_side)
        enkelt_feil_liste = np.zeros(N_eval)
        for j in range(len(losning)):
            enkelt_feil_liste[j] = abs(losning[j] - rho_analytisk[j])
        errorliste[i] = np.linalg.norm(enkelt_feil_liste,np.inf)
    return errorliste

#print(error())

plt.figure()
plt.loglog(lambda_liste,error(d1,lambda_liste))
plt.xlabel('Error mot lambda med d=0.25')
plt.show()

plt.figure()
plt.loglog(lambda_liste,error(d2,lambda_liste))
plt.xlabel('Error mot lambda med d=2.5')
plt.show()