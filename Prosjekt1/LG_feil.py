import pickle

import matplotlib.pyplot as plt
import numpy as np
from Vitber.Prosjekt1.Vår_kode.F_analytisk import analytical_solution

from Vitber.Prosjekt1.metoder import chebyshev, fredholm_lhs, rho, fredholm_rhs

a = 0
b = 1
    # d is the distance from the measurements in the kernel
d = 2.5E-02
    # we use rho(y) = exp(gamma*y)*sin(omega*y) as an example
gamma = -2
omega = 3*np.pi
# maximal order of the Taylor series expansion
Nmax  = 75
    # evaluate the integral expression at x_eval
N_eval = 40
Nq = 40
xq,w = np.polynomial.legendre.leggauss(Nq)
t = 0.5*(xq + 1)*(b - a) + a
xc = chebyshev(a,b,N_eval)
xs = chebyshev(a,b,N_eval)

F = analytical_solution(a,b,omega,gamma,Nmax)
F = pickle.load( open( "F.pkl", "rb" ) )
F_eval = F(xc,d)

A_ganger_rho = np.matmul(fredholm_lhs(xc,xs,t,w/2),rho(omega,gamma,xs,N_eval))


def error_lg(Nq_liste,a,b):
    #error = np.zeros(Nq)
    error = np.zeros(len(Nq))
    F = fredholm_rhs(xc,F_eval)
    for i in Nq_liste:
        xq,w = np.polynomial.legendre.leggauss(i)
        t = 0.5*(xq + 1)*(b - a) + a
        A_rho = np.matmul(fredholm_lhs(xc,xs,t,w/2),rho(omega,gamma,xs,N_eval))
        ny_liste = np.zeros(N_eval)
        for j in range(N_eval):
            ny_liste[j] = F[j] - A_rho[j]
        error[i-1] = np.linalg.norm(ny_liste,np.inf)
    return error


Nq = np.arange(10,50,1)
error_list = error_lg(Nq,a,b)

plt.figure()
plt.title('F(x)')
plt.plot(xc,F_eval)
plt.xlabel('$x$')
plt.show()

plt.figure()
plt.plot(xc,A_ganger_rho)
plt.xlabel('$x$')
plt.show()

plt.figure()
plt.title('Error med Legendre Gauss')
plt.plot(Nq,error_list)
plt.plot('Nq')
plt.show()