
    def isLegalTwist(grid, x, clockwise): #Check if twist is legal

        if  !(1 < x < self.L) or (grid[int(np.round(n / 2)), int(np.round(n / 2 - self.L / 2)) + (x-1)] == 0):
            return False
        if clockwise == True:   # Alle bortsett fra aminosyren helt til høyre hopper oppå den neste, aminosyren helt til hyre hopper under den forje
            if (grid[int(np.round(n / 2)), int(np.round(n / 2 - N / 2)) + x] != 0)  #sjekker om det er siste aminosyren
                if
            else
        else if

        return True/False



def matrixExpansion(grid, x):
    col, row = findX(grid, x)
    width, height = np.shape(grid)
    print(width, height)
    if ( width % 2 != 0):
        midt = math.ceil(width/2)
        if x > midt:
            for i in col-1:
                utvid=np.insert(grid,width+i,0,axis=0)
        else:
            for i in width-1:
                utvid=np.insert(grid,0-i,axis=0)

    else:           # partallsmatrise
        if x>=width/2: #befinner seg over midten
            for i in col-1:
                utvid=np.insert(grid,width+i,0,axis=0)
        else:
            for i in

    if ( height % 2 != 0):
        midt = math.ceil(height/2)
        if x > midt:
            utvidK = col - 1
        else:
            utvidK = midt - col
    else:
        utvidK = col -1
    new = np.zeros((width + utvidR , height + utvidK)).astype(np.int16)
    print(new)
    #result[int(np.round(len(grid)/2)),int(np.round(len(grid)/2 )]


def twist(grid, x, clockwise):


def twist1(grid, x, clockwise):
   # if isLegalTwist(grid, x, clockwise) == True:
    col, row = findX(grid, x)
    if clockwise == True:
        rotate = grid[col:len(grid), row:len(grid)]
        rotated = rotateMatrix(rotate, True)
        #Sett på plass og sjekk om det kræsjer
    else:
        rotate = grid[0:col, 0: row]
        rotated = rotateMatrix(rotate, False)
        # Sett på plass og sjekk om det kræsjer


        #return twistedGrid

