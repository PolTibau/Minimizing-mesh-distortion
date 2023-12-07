import numpy as np
import matplotlib.pyplot as plt

from mesh            import X,T,Nint,plotMesh,dofsToCoords,coordsToDofs
from differentiation import derivadaNumerica,hessianaNumerica
from distortion      import calculaDistorsioMalla
from newtonraphson_practica2 import iteracio
#------------------------------------------------------------------------------
#
# Completar el codi per determinar la posicio dels nodes interiors que
# minimitza la distorsio de la malla. 
# Abans de fer servir el metode de Newton, cal completar la funcio
# calculaDistorsioMalla
#  ...
#  ...
#  ...
#
#------------------------------------------------------------------------------

# Plot initial mesh configuration
plotMesh(X,'Initial mesh')

res = calculaDistorsioMalla(X, T)
print('Distorsio inicial: ',res)

# Convert your mesh coordinates to a vector with the dofs
y = coordsToDofs(X)


def iteracio(I,y,R,E):
    R = derivadaNumerica(F,y)
    J = hessianaNumerica(F,y)
    I = np.linalg.solve(J,-R)
    y += I
    Cond.append(np.linalg.cond(J))
    E.append(np.log10((np.linalg.norm(I)/np.linalg.norm(y))))
    return R,E

def F(y):
    return calculaDistorsioMalla(dofsToCoords(y),T)

# We want to minimize F(y) like if it was a zero function problem

tol = 0.5*1e-7
R = derivadaNumerica(F,y)
I = np.zeros(R.shape)
E = []
Cond = []
print(X[0][:])
nomb_iter = 0
#we have to compute a 0 of dy using Newton-Raphson
while(abs(R.all())>tol):
    R, E = iteracio(I, y, R, E)
    nomb_iter += 1

# Convert you vector with the dofs back to mesh coordinates
X = dofsToCoords(y)
plotMesh(X,'Final mesh')
# X = dofsToCoords(np.ones(y.shape))
#We can observe that final mesh has triangles "more equilater"
distf = calculaDistorsioMalla(X, T)
print('Distorsio final: ',distf)
print('Posició del primer node interior final és: ', X[0][:])
print("El nombre d'iteracions necessari per un error menor de 0.5*1e-7 és: ", nomb_iter)
print(E)
print(Cond)

# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
x = np.arange(11)
plt.yticks(np.arange(0, 0.25, step=0.01),labels=None,minor=True, fontsize=5)  # Set label locations.
plt.plot(x,E, 'ro-')  # Plot some data on the axes.
plt.xlabel("Iteració")
plt.ylabel("log10 Error relatiu")
for y,z in zip(x,E):
    label = "{:.2f}".format(z)
    plt.annotate(label, # this is the text
                 (y,z), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(10,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.legend()  # Add a legend.
plt.show()
