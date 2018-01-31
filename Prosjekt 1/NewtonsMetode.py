# Solve a - 1/x = 0 using Newton's method
# Does not require any divisions
# Note: we cannot even evaluate the residual w/o divisions!
# However one could e.g. check |a*x-1|

a = 3.0

x0 = 2.0
#x0 = 0.5
#x0 = 2./3

# in practical situations of course we do not know the error!
e0 = abs(x0-1/a)
for i in range(15):
    x1 = x0*(2-a*x0)
    e1 = abs(x1-1/a)
    print('Iter: %3d, x: %15.10e, |x*a-1|: %e, e1/e0: %e' %(i, x1, abs(x1*a-1), e1/e0))
    if abs(x0-x1)/max(1.0E-06,abs(x0)) < 1.0E-09:
        # Success!
        print('\nConvergence!')
        break
    x0 = x1
    e0 = e1