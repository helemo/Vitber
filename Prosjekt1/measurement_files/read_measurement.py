import numpy as np

file_name = 'q8_test.npz'
f         = open(file_name,'rb')
npzfile   = np.load(f)

print(npzfile.files)

print(npzfile['a'], npzfile['b'], npzfile['d'])
print(npzfile['xc'])
print(npzfile['F'])