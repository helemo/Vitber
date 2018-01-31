import math

# bisection method for solving the equation
# f(x) = 0
# on the interval [a,b]
# we assume a<b, f(a)*f(b) < 0
# we stop the method when we have bracketed the root
# up to the tolerance tol
def bisect(f,a,b,tol):
    assert(a<b)
    assert(tol>0.)
    fa = f(a)
    fb = f(b)
    assert(fa*fb<0.)

    # iteration number
    iter = 0
    while b-a > tol :
        iter = iter + 1
        c  = (a+b)/2.0
        fc = f(c)
        print("Iter: %4d, a: %e, f(a): %e, c: %e, f(c): %e, b: %e, f(b): %e" % (iter,a,fa,c,fc,b,fb))

        if fa*fc < 0 :
            b  = c
            fb = fc
        else:
            a  = c
            fa = fc

    return c

# example equation: what is the square root of 0.5?
def f1(x):
    return math.cos(x) - x

# test our method
if __name__=='__main__':
    a = 0
    b = 1
    tol = 1.0E-06
    c = bisect(f1,a,b,tol)