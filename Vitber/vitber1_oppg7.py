#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:44:34 2018

@author: ninabg
"""

import numpy as np
import sympy as sy
import scipy.special as ss
import pickle
import time
import matplotlib.pyplot as plt
from vitber1_analytiskF import analytical_solution
from vitber1_funksjoner import chebyshev,Lagrange,rho,K,Newton_Cotes,fredholm_lhs,fredholm_rhs

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
F_eval = F(xc,d1)

xq_old, w = np.polynomial.legendre.leggauss(N_eval**2)
xq = 0.5*(xq_old + 1)*(b - a) + a
A = fredholm_lhs(xc, xs, xq, w/2)
B = fredholm_rhs(xc, F_eval)

def finn_b(d):
    F_eval = F(xc,d)
    F_error = np.zeros(len(xc))

    for i in range(len(xc)):
        F_error[i] = F_eval[i]*(1+np.random.uniform(-delta,delta,N_eval)[i])
    return F_error
    


def error():
    A = fredholm_lhs(xc, xs, xq, w/2)
    A_t = np.transpose(A)
    venstre_side = np.add(np.matul(A_t,A),
    rho = np.linalg.solve(np.matul(A_t,A),B)
    errorliste = np.zeros(30)
    lambda_liste = np.geomspace(10**(-14),10,N_eval)
    for i in range(N_eval):
        venstre_side = np.add(np.matul(A_t,A),np.multiply(lambda_liste[i],np.eye))
        hoyre_side = np.matmul(A_t,B)
        rho = np.linalg.solve(np.matul(A_t,A),B)
        enkelt_feil_liste = np.zeros(N_eval)
        for j in range(len(rho)):
            enkelt_feil_liste[j] = abs(rho[j] - rho_analytisk[j])
        errorliste[i] = np.linalg.norm(enkelt_feil_liste,np.inf)
    return errorliste
    
    



