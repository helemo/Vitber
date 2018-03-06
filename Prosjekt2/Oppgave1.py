import scipy
import numpy as np
import math
from Vitber.Prosjekt2.expandMatrix import expandMatrix

def makeGrid(n):           #nxn matrise
    grid = np.zeros((n,n)).astype(np.int16)
    return grid

def findX(grid, x):
    col, row = np.where( grid == x)
    col = int(col)
    row = int(row)
    return col,row


class Polymer:  #polymer med lengde L i nxn grid
    def __init__(self, L, grid):
        self.L = L
        self.grid = grid
        assert(L < len(grid))
        n = len(grid)
        grid[int(np.round(n / 2)), int(np.round(n / 2 - L / 2)): int(np.round(n / 2 - L / 2)) + L] = np.linspace(1, L,
                                                                                                                  L).astype(
            np.int16)

    def sum(self):
        def summarize(x):
            if x == 0:
                return 0
            else:
                return (x + summarize(x-1))
        return summarize(self.L)


    def getGridLength(self):
        return len(self.grid)

    def isLegalTwist(self, grid, x, clockwise):
        pass

    def Twist(self, grid, x, clockwise):
        if self.isLegalTwist(grid, x, clockwise):
            pass
        else:
            #Try again
            pass
        return


def isLegalTwist(grid, x, clockwise):
    original = grid
    copy = copyGrid(grid, x)        # Må prøve å fjerne tall på begge sider av polymeret, så må en try her inn et sted
    expanded = expandMatrix(copy, x)  # Må rotere uten x pga indekser
    if clockwise == True:
        rotated = np.rot90(expanded,1,(1,0))
    else:
        rotated = np.rot90(expanded,1 ,(0,1))
    result = compareMatrices(original, rotated)
    return result


'''''
def compareMatrices(original, rotated):
    result = original + rotated
    if np.sum(original) == np.sum(result):
        return result
    else:
        # Kast en exception og try again??  # TODO
'''''


def copyGrid(grid, x):
    N = 3       # OBS!!
    #col, row = findX(grid, x)
    copyGrid = grid
    for row in copyGrid:
        for col in copyGrid:
            if x < round(N / 2):
                if ( np.where( copyGrid[row, col] >= x) ):
                    copyGrid[row, col] = 0
            else:
                if ( np.where( copyGrid[row, col]  <= x) ):
                    copyGrid[row, col] = 0
    return copyGrid