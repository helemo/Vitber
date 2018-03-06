import numpy as np
import math
from Vitber.Prosjekt2.Oppgave1 import *


def twistTest(grid, x, clockwise):
   original = grid
   print(original, '1')
   copy = copyGrid(grid, x)  #må fikses
   print(copy, '2')
   #expanded = expandMatrix(copy, x)
   #print(expanded, '3')
   if clockwise == True:
       rotated = np.rot90(copy,1 , (1,0))
       print(rotated, 'rot')
   result = rotated + original
   print(result, '4')

mat =    [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 1, 2, 3, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]

print(mat[3,1])
for row in mat:
    for col in mat:
        if mat[row, col] < 3:
            mat[row, col] = 0
#print(mat)

#mat = np.array(mat)
#print(mat)

#twistTest(mat, 3, True)


def copyGrid(grid, x):
    N = 3       # OBS!!
    #col, row = findX(grid, x)
    copyGrid = grid
    for row in copyGrid:
        for col in copyGrid:
           if ( np.where( copyGrid[row,col] < x)):
               copyGrid[row,col] = 0
    return copyGrid

#print(copyGrid(mat, 3))

