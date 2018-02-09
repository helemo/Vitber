import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,1,100)

# Setter inn grenseverdiene rho i funksjonen K

def K(d,x):
    return (2/3-x)/(d*np.sqrt(d**2+(x-2/3)**2))-(1/3-x)/(d*np.sqrt(d**2+(x-1/3)**2))

plt.xlim(0,1)
plt.semilogy(x,K(0.025,x))
plt.semilogy(x,K(0.25,x))
plt.semilogy(x,K(2.5,x))
plt.legend(['d = 0.025','d = 0.25','d = 2.5'])
plt.xlabel('x')
plt.ylabel('F(x)')
plt.xlim(0,1)
plt.show()