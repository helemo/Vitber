import numpy as np
import math

def findX(grid,x):
    col, row = np.where( grid == x)
    col = int(col)
    row = int(row)
    return col,row

def expandMatrix(grid, x):
    xcol , xrow = findX(grid, x)
    col = grid.shape[0]
    row = grid.shape[1]

    i = j = 0
    row_right = row - xrow - 1
    row_left = row - row_right - 1
    col_down = col - xcol - 1
    col_up = col - col_down - 1
    if col % 2 != 0:  # Kolonner er oddetall
        # if xrow == math.floor(row / 2): # x er i midten mhp rows
        if xrow > math.floor(row / 2):  # x er til HØYRE for midten mhp rows
            for i in range(0, row_left - row_right):
                grid = np.insert(grid, row, 0, axis=1)
                i += i
            if xcol > col / 2:  # x er i nedre, høyre kvadrant
                for j in range(0, col_up - col_down):
                    grid = np.insert(grid, col, 0, axis=0)
                    j += j
            elif xcol < col / 2:  # øvre, høyre
                for j in range(0, col_down - col_up):
                    grid = np.insert(grid, 0, 0, axis=0)
                    j += j

        elif xrow < math.floor(row / 2):  # x er til VENSTRE mhp rows
            for i in range(0, row_right - row_left):
                grid = np.insert(grid, 0, 0, axis=1)
                i += i
            if xcol < col / 2:  # øvre, venstre
                for j in range(0, col_down - col_up):
                    grid = np.insert(grid, 0, 0, axis=0)
                    j += j
            elif xcol > col / 2:  # nedre, venstre
                for j in range(0, col_up - col_down):
                    grid = np.insert(grid, col, 0, axis=0)
                    j += j

        else:  # x på midten mhp rows
            if xcol < col / 2:  # øvre
                for j in range(0, col_down - col_up):
                    grid = np.insert(grid, 0, 0, axis=0)
                    j += j
            elif xcol > col / 2:  # nedre
                for j in range(0, col_up - col_down):
                    grid = np.insert(grid, col, 0, axis=0)
                    j += j

    else:  # Kolonner er partall
        if xrow >= row / 2:  # x er til HØYRE
            for i in range(0, row_left - row_right):
                grid = np.insert(grid, row, 0, axis=1)
                i += i
            if xcol >= col / 2:  # x er i nedre, høyre kvadrant
                for j in range(0, col_up - col_down):
                    grid = np.insert(grid, col, 0, axis=0)
                    j += j
            else:  # øvre, høyre
                for j in range(0, col_down - col_up):
                    grid = np.insert(grid, 0, 0, axis=0)
                    j += j

        elif xrow < row / 2:  # x er til VENSTRE
            for i in range(0, row_right - row_left):
                grid = np.insert(grid, 0, 0, axis=1)
                i += i
            if xcol < col / 2:  # øvre, venstre
                for j in range(0, col_down - col_up):
                    grid = np.insert(grid, 0, 0, axis=0)
                    j += j
            else:  # nedre, venstre
                for j in range(0, col_up - col_down):
                    grid = np.insert(grid, col, 0, axis=0)
                    j += j

                    # Print ut output
                    # print('ny grid =\n', grid)

    return grid