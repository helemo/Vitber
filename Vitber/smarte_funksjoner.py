# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:53:56 2018

@author: mariadahle
"""

import numpy as np

def Lagrange(xs,xq,j):
    Lj = 1
    for m in range(len(xs)):
        if m != j:
                Lj *= (xq-xs[m])/(xs[j]-xs[m])
    return Lj



def rho(omega,gamma,xs,Ns):
    rho = np.ones(Ns)
    for i in range(Ns):
        rho[i] = np.sin(omega*xs[i])*np.exp(gamma*xs[i])
    return rho
    


def K(x,y,d):
    return d/(d**2+(y-x)**2)**(3/2)
    

def w_k(a,b,Nq):
    w = (a-b)/Nq*np.ones(Nq)
    return w


def Newton_Cotes(a,b,Nq):
    step = ((b-a)/Nq)
    I = np.arange(0, Nq, 1)
    X = (a + step*I)
    w = (b-a)/Nq*np.ones(Nq)
    return X,w
 


def fredholm_lhs(xc, xs, xq, w):
    Nc = xc.shape[0]
    Ns = xs.shape[0]
    A = np.zeros((Nc,Ns))
    for i in range(Nc):
        for j in range(Ns):
            #A[i][j] = np.sum(K(xc[i],xq,0.025)*Lagrange(xs,xq,j)*w[k]
            A[i][j]=0
            for k in range(len(xq)):
                A[i][j] += K(xc[i],xq[k],0.025)*Lagrange(xs,xq[k],j)*w[k]
    return A


def fredholm_rhs(xc,F):
    Nc = xc.shape[0]
    b = np.zeros(Nc)
    for i in range(Nc):
        b[i] = F[i]
    return b


def chebyshev(a, b, N):
    # return Chebyshev's interpolation points on the interval [a,b]
    I = np.arange(1, N+1, 1)
    X = (b + a)/2 + (b - a)/2*np.cos((2*I - 1)*np.pi/(2*N))
    return X