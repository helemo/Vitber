# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:09:01 2018

@author: mariadahle
"""

#determinanten skal bli A
import matplotlib.pyplot as plt
import numpy as np

def A(x):
    return np.array([[1,2,3,x],[4,5,x,6],[7,x,8,9],[x,10,11,12]])

def f(y):
    return np.linalg.det(A(y)) - 1000

#plotter funksjonen
X = np.linspace(9.5, 10, 100)
plt.plot(X, [f(z) for z in X])
plt.xlim(X[0],X[-1])
plt.grid(True)
plt.draw()

#skal nå finne f(x) = 0, som er bisection method
#input er a og b sånn at f(a)f(b) < 0

tol = 1.0e-6 #skal ha nøyaktighet på 6 desimaler
tol_fun = 1.0e-20
a = 9.5
b = 10.

fa=f(a)
fb=f(b)
assert(fa*fb<0)
i = int(0)
while b-a>tol:
    i += 1
    c = (b+a)/2
    fc = f(c)
    print('Iter: %4d, a: %e, f(a): %e, c: %e, f(c): %e, b: %e, f(b): %e \n' % (i,a,fa,c,fc,b,fb))
    if abs(fc) < tol_fun:
        break
    if fa*fc<0:
        b = c
        fb = fc
    else:
        a = c
        fa = fc

print('Backward error: %e' % fc)
    
plt.show()
   
    
    
    
    
    
    

