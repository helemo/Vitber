# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:35:51 2018

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


#Integralet vi skal løse: int_a^b K(x,y) rho(y) dy
    
a = 0
b = 1
d = 2.5E-02
# we use rho(y) = exp(gamma*y)*sin(omega*y) as an example
gamma = -2
omega = 3*np.pi
# maximal order of the Taylor series expansion
Nmax  = 75
# evaluate the integral expression at x_eval
N_eval = 40
print('Hei')


#Nqq = 40
#Nq_eval = np.arange(1,50,1)
#xq,w = Newton_Cotes(a,b,Nqq)

#print(Lagrange([1,2,3],3/2,0))

#print(len(fredholm_lhs(xc,xs,xq,w)))

#A_ganger_rho = np.matmul(fredholm_lhs(xc,xs,xq,w),rho(omega,gamma,xs,N_eval))
#print(F_eval)
#print(A_ganger_rho)
#print(F_eval)


def error(Nq_liste,a,b):
    #error = np.zeros(Nq)
    error = np.zeros(len(Nq_liste))
    F = fredholm_rhs(xc,F_eval)
    for i in range(len(Nq_liste)):
        xq,w = Newton_Cotes(a,b,Nq_liste[i])
        A_rho = np.matmul(fredholm_lhs(xc,xs,xq,w),rho(omega,gamma,xs,N_eval))
        ny_liste = np.zeros(N_eval)
        for j in range(N_eval):
            ny_liste[j] = F[j] - A_rho[j]
        error[i] = np.linalg.norm(ny_liste,np.inf)
    return error



def error_lg(Nq_liste,a,b):
    #error = np.zeros(Nq)
    error = np.zeros(len(Nq_liste))
    F = fredholm_rhs(xc,F_eval)
    for i in range(len(Nq_liste)):
        xq,w = np.polynomial.legendre.leggauss(Nq_liste[i])
        t = 0.5*(xq + 1)*(b - a) + a
        A_rho = np.matmul(fredholm_lhs(xc,xs,t,w/2),rho(omega,gamma,xs,N_eval))
        ny_liste = np.zeros(N_eval)
        for j in range(N_eval):
            ny_liste[j] = F[j] - A_rho[j]
        error[i] = np.linalg.norm(ny_liste,np.inf)
    return error


xc = chebyshev(a,b,N_eval)
xs = chebyshev(a,b,N_eval)



#F = analytical_solution(a,b,omega,gamma,Nmax)
start_t= time.time()
F = pickle.load( open( "F.pkl", "rb" ) )
end_t  = time.time()
print("Initialization took %f s." % (end_t-start_t))
F_eval = F(xc,d)

Nq = np.arange(20,50,1)

start_t = time.time()
error_list_lg = error_lg(Nq,a,b)
error_list = error(Nq,a,b)
end_t = time.time()
print('Det tok %f sek å finne erroren' % (end_t-start_t))
"""
xq,w = Newton_Cotes(a,b,40)
A_ganger_rho = np.matmul(fredholm_lhs(xc,xs,xq,w),rho(omega,gamma,xs,N_eval))
b = fredholm_rhs(xc,F_eval)
print(A_ganger_rho)
print(b)
"""
#plotter F og A*rho
"""
plt.figure()
plt.plot(xc,F_eval)
plt.plot(xc,A_ganger_rho)
plt.title('F(x)')
plt.xlabel('x')
plt.show()


plt.figure()
plt.plot(xc,A_ganger_rho)
plt.title('A*rho')
plt.xlabel('x')
plt.show()
"""
print('hei igjen')


#plotter error
plt.figure()
plt.title('Error')
plt.plot(Nq,error_list,'k')
plt.plot(Nq,error_list_lg)
plt.xlabel('Nq')
plt.show()


plt.figure()
plt.title('Logaritmisk skalert error')
plt.semilogy(Nq,error_list,'k')
plt.semilogy(Nq,error_list_lg)
plt.xlabel('Nq')
plt.show()










