import math
import matplotlib.pyplot as plt
import numpy as np

#
# Fixed-point iteration for finding a square root, i.e. solving the equation
# x*x - a = 0 for a given a
#
a = 0.5


# equation plus derivatives
def f(x):  return x * x - a


def df(x):  return 2 * x


def d2f(x): return 2.0


# We have several possibilities for a fixed-point iteration, e.g.:
version = 3

if version == 0:
    # stupid version
    def g(x):
        return a / x


    def dg(x):
        return -a / (x * x)
elif version == 1:
    # Richardsson's iteration
    w = .4


    def g(x):
        return x - w * (x ** 2 / a - 1)


    def dg(x):
        return 1 - w * 2 * x / a
elif version == 2:
    # Newton's iteration/Babylonian/Heron's method
    def g(x):
        return 0.5 * (x + a / x)


    def dg(x):
        return 0.5 * (1 - a / x ** 2)
elif version == 3:
    # Chebyshev's iteration
    def g(x):
        return 0.375 * x + 0.75 * a / x - 0.125 * a ** 2 / x ** 3


    def dg(x):
        return 0.375 - 0.75 * a / x ** 2 + 0.375 * a ** 2 / x ** 4

# Plot g vs x first
X = np.linspace(0.4, 1, 200)
plt.plot(X, X, 'b', label='x')
plt.plot(X, g(X), 'r', label='g(x)')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.draw()

# Run the fixed-point iteration and check convergence
N = 100
tol = 1.0E-08
iter = 0
# initial guess
x0 = 1.0
# in reality we of course do not know this
# -otherwise we would not have to solve the equation in the first place!
r = np.sqrt(a)
e0 = abs(x0 - r)

while iter < N:
    iter = iter + 1
    x1 = g(x0)
    e1 = abs(x1 - r)

    print('Iter: %3d, x: %15.10e, f(x): %15.10e, e1/e0: %e' % (iter, x1, f(x1), e1 / e0))

    # in practice it is a good idea to check the residual |f(x)| here as a stopping criterion, i.e.
    # if abs(f(x1)) < tol:
    if abs(x1 - x0) / max(1.0, abs(x0)) < tol:
        # success, converged!
        print('\nConvergence!')
        print('Iter: %3d, x: %e, f(x): %e, g\'(x): %e\n' % (iter, x1, f(x1), dg(x1)))
        break
    x0 = x1
    e0 = e1

plt.show()