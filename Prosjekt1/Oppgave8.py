import matplotlib.pyplot as plt
import numpy as np

# from import analytical_solution
from Vitber.Prosjekt1.metoder import fredholm_lhs, Legendre, Chebyshev


def Reconstruct_Density(file, K):
    f = open(file, 'rb')
    npzfile = np.load(f)
    xs = Chebyshev(len(npzfile['xc']), npzfile['a'], npzfile['b'])
    xq, w = Legendre(len(npzfile['xc']**2), npzfile['a'], npzfile['b'])
    A = fredholm_lhs(npzfile['xc'], xs, xq, w, K, npzfile['d'])
    r = [npzfile['xc']]
    for i in range(-14, 2):
        print(i)
        lhs = np.dot(A.T, A) + np.dot(10**i, np.identity(len(npzfile['xc'])))
        rhs = np.dot(A.T, npzfile['F'])
        p = np.linalg.solve(lhs, rhs)
        plt.plot(npzfile['xc'], p)
        r.append(p)
        #plt.plot(npzfile['xc'], y, label = 'The function y=sin(5*Pi*x)')
        plt.legend()
        plt.show()
    return r

K = lambda x, y, d:  d * (d**2 + (y-x)**2)**(-3/2)
r1 = Reconstruct_Density('q8_1.npz', K)
r2 = Reconstruct_Density('q8_2.npz', K)
r3 = Reconstruct_Density('q8_3.npz', K)
for i in range(1, len(r1)):
    print(-15+i)
    plt.plot(r1[0], r1[i], label = 'First file')
    plt.plot(r2[0], r2[i], label = 'Second file')
    plt.plot(r3[0], r3[i], label = 'Third file')
    plt.legend()
    plt.show()