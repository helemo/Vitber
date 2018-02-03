# plot Lagrange polynomials on [a,b]

import numpy as np
import matplotlib.pyplot as plt

a = -1
b =  1
N =  7


# points for interpolation
Xb = np.linspace(a,b,N)
# plot the interpolation points
fs = 24 #font size
plt.plot(Xb,np.zeros_like(Xb),'x',linewidth=3,markeredgewidth=3,markersize=10)
plt.grid(True)
plt.xlabel('x', fontsize=fs)
plt.ylabel('y', fontsize=fs)
plt.show(block=False)
# wait for a key press
_ = input("Press [enter] to continue.")

# points for visualization
Xf = np.linspace(a,b,100)

cm = plt.cm.gist_ncar
colors = [cm(i) for i in np.linspace(0,1,N+1)]

for i in range(N):
    # form the Lagrange polynomial i
    Li = lambda x: np.prod(x-Xb[np.arange(N)!=i])/np.prod(Xb[i]-Xb[np.arange(N)!=i])
    # plot it
    plt.plot(Xf,[Li(x) for x in Xf], '-', Xb,[Li(x) for x in Xb],'o',linewidth=3,markeredgewidth=3,markersize=10,color=colors[i])
    plt.show(block=False)
    plt.draw()
    # wait for a key press
    if i<N:
        _ = input("Press [enter] to continue.")

plt.show(block=True)