import numpy as np

def read_measurement():
    file_name = 'q8_1.npz'
    f         = open(file_name,'rb')
    npzfile   = np.load(f)

    print(npzfile.files)

    print(npzfile['a'], npzfile['b'], npzfile['d'])
    print(npzfile['xc'])
    print(npzfile['F'])
    return npzfile['a'], npzfile['b'], npzfile['d'], npzfile['xc'], npzfile['F']


a, b, d, xc, F = read_measurement()
