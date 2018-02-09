# -*- coding: utf-8 -*-

import numpy as np
from math import *


def Lagrange(xs, xq, j):
    Lj = 1
    for m in range(len(xs)):
        if m != j:
            Lj *= (xq - xs[m]) / (xs[j] - xs[m])
    return Lj


def rho(omega, gamma, xs, Ns):
    rho = np.ones(Ns)
    for i in range(Ns):
        rho[i] = np.sin(omega * xs[i]) * np.exp(gamma * xs[i])
    return rho


def K(x, y, d):
    return d / (d ** 2 + (y - x) ** 2) ** (3 / 2)


def w_k(a, b, Nq):
    w = (a - b) / Nq * np.ones(Nq)
    return w


def Newton_Cotes(a, b, Nq):
    step = ((b - a) / Nq)
    I = np.arange(0, Nq, 1)
    X = (a + step * I)
    w = (b - a) / Nq * np.ones(Nq)
    return X, w


def fredholm_lhs(xc, xs, xq, w):
    Nc = xc.shape[0]
    Ns = xs.shape[0]
    A = np.zeros((Nc, Ns))
    for i in range(Nc):
        for j in range(Ns):
            # A[i][j] = np.sum(K(xc[i],xq,0.025)*Lagrange(xs,xq,j)*w[k]
            A[i][j] = 0
            for k in range(len(xq)):
                A[i][j] += K(xc[i], xq[k], 0.025) * Lagrange(xs, xq[k], j) * w[k]
    return A


def fredholm_rhs(xc, F):
    Nc = xc.shape[0]
    b = np.zeros(Nc)
    for i in range(Nc):
        b[i] = F[i]
    return b


def chebyshev(a, b, N):
    # return Chebyshev's interpolation points on the interval [a,b]
    I = np.arange(1, N + 1, 1)
    X = (b + a) / 2 + (b - a) / 2 * np.cos((2 * I - 1) * np.pi / (2 * N))
    return X

#For the last problem it seemed like it might be relevant to have the option to generate Chebyshev's interpolation
#nodes in the general range a, b instead of the specific interval [0, 1], so this method is meant to do this.
def Chebyshev(n, a, b):
    r = []
    for i in range(1, n+1):
        r.append(0.5*(a+b)+0.5*(a-b)*cos(pi*(2*i-1)/(2*n)))
    return r

#Returns nodes and weights for the legendre gauss quadrature in the generic interval [a, b] for use in problem 8.
def Legendre(n, a, b):
    x1, w1 = np.polynomial.legendre.leggauss(n)
    xq = [((b-a)*x+b+a)/2 for x in x1]
    w = [0.5*(b-a)*x for x in w1]
    return xq, w
