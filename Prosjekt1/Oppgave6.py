import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

from Vitber.Prosjekt1.metoder import chebyshev, rho, fredholm_lhs, fredholm_rhs

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



analytisk_rho = rho(omega,gamma,xs,N_eval)
xq_old, w = np.polynomial.legendre.leggauss(N_eval**2)
xq = 0.5*(xq_old + 1)*(b - a) + a
A = fredholm_lhs(xc, xs, xq, w/2)
B = fredholm_rhs(xc, F_eval1)
losning = np.linalg.solve(A,B)


# analytisk rho uten perturbering
rho1 = np.linalg.solve(A,F_eval1)
rho2 = np.linalg.solve(A,F_eval2)
rho3 = np.linalg.solve(A,F_eval3)

# analytisk rho med perturbering
rho1_error = np.linalg.solve(A,F_error1)
rho2_error = np.linalg.solve(A,F_error2)
rho3_error = np.linalg.solve(A,F_error3)


plt.figure('F og F med feil')
plt.plot(xc,F_eval1, label='F ved d = 0.025')
plt.plot(xc,F_error1, label='F med feil ved d = 0.025')
plt.plot(xc,F_eval2, label='F ved d = 0.25')
plt.plot(xc,F_error2, label='F med feil ved d = 0.25')
plt.plot(xc,F_eval3, label='F ved d = 2.5')
plt.plot(xc,F_error3, label='F med feil ved d = 2.5')
plt.legend()
plt.title('b')
plt.show()


plt.figure()
plt.plot(xs,analytisk_rho)
plt.title('Analytisk rho')
plt.show()



plt.figure()
plt.plot(xs, rho1, label='Rho ved d = 0.025')
plt.plot(xs, rho2, label='Rho ved d = 0.25')
plt.plot(xs, rho3, label='Rho ved d = 2.5')
plt.legend()
plt.title( r'Analytisk $\rho$ uten perturbering')
plt.show()

plt.figure()
plt.title( r'Feil i analytisk $\rho$ med perturbering')
plt.plot(xs, rho1_error, label='Rved d = 0.025')
plt.plot(xs, rho2_error, label='Rho ved d = 0.25')
plt.plot(xs, rho3_error, label='Rho ved d = 2.5')
plt.legend()
plt.show()





