import numpy as np
import time

size = 4
nPolice = 1
nThief = 1
nGold = 1

mySize = [4]
mySize *= 2*(nPolice+nThief+nGold)

a = np.random.rand(2,2)
print(a)


a = a*(2*size) - size
print(a)