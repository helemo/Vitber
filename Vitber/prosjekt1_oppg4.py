# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:37:45 2018

@author: mariadahle
"""
import numpy as np
import matplotlib.pyplot as plt

def lagrange(i,x0,xs):
    Li = lambda x: np.prod(x0-xs[np.arange(len(xs))!=i])/np.prod(xs[i]-xs[np.arange(len(xs))!=i])
    return Li

def chebyshev(a, b, N):
    # return Chebyshev's interpolation points on the interval [a,b]
    I = np.arange(1, N+1, 1)
    X = (b + a)/2 + (b - a)/2*np.cos((2*I - 1)*np.pi/(2*N))
    return X

