import numpy as np

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

''' Input dataset'''

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
print(X)
from tabulate import tabulate
#print(tabulate(X))

''' Output dataset'''

y = np.array([[0, 0, 1, 1]]).T
#print(tabulate(y))

''' Initialize weights randomly with mean 0'''

np.random.seed(1)
syn0 = 2 * np.random.random((3, 1)) - 1
#print(tabulate(syn0))

''' Training loop'''

L1error=[]

for iter in range(10000):
    
    ''' Forward propagation'''

    l0 = X

    l1 = nonlin(np.dot(l0, syn0))

    ''' Calculate error'''
    
    l1_error = y - l1

    L1error.append(l1_error)

    

    ''' Update weights'''
    
    l1_delta = l1_error * nonlin(l1, True)
    syn0 += np.dot(l0.T, l1_delta)
#print(tabulate(syn0))

# Print final output
print("Output After Training:")
print(l1)
print(tabulate(L1error))
