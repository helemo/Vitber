from math import *
from matplotlib import pyplot as plt
import numpy as np

def Y (d):
    y = []
    for i in range (101):
        x = i/100
        y.append((2/3-x)/(d*(d**2+(x-2/3)**2)**(1/2))-(1/3-x)/(d*(d**2+(x-1/3)**2)**(1/2)))
    return y

def X ():
    x = []
    for i in range (101):
        x.append(i/100)
    return x

def PlotXY (x, y):
    for i in range (len(y)):
        plt.plot(x, y[i])
    plt.semilogy()
    plt.show()


def main():
    d = [0.025, 0.25, 2.5]
    y = []
    for i in range(len(d)):
        y.append(Y(d[i]))
    x = X()

    PlotXY(x, y)

main()