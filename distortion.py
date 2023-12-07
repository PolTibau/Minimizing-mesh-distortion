import numpy as np
import matplotlib.pyplot as plt
from mesh import X,T,Nint

A = np.array([[1, -np.sqrt(3.) / 3], [0, 2 * np.sqrt(3.) / 3]],dtype=float)


def calculaDistorsioMalla(X, T):
    suma = 0
    for i in range(np.shape(T)[0]):
        D = np.zeros((2,2))
        D[0] = X[T[i][1]]-X[T[i][0]]
        D[1] = X[T[i][2]]-X[T[i][0]]
        d_phi = np.matmul(np.transpose(D),A)
        suma += ((np.linalg.norm(d_phi, 'fro')**2)/(2*abs(np.linalg.det(d_phi))))**2
    dist = np.sqrt(suma)
    return dist
 
distorsio = calculaDistorsioMalla(X, T)
print(distorsio)        
