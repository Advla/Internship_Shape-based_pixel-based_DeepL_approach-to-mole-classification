import os
import matplotlib.pyplot as plt
import numpy as np 
import skfda 
from scipy.spatial import procrustes
os.chdir('C:/Users/andre/Desktop/Stage 2A/Projet/code_contours_issam/')
# Reading image 

## Fourier basis expansion function

def get_basis (C1, C2, M=11):     
    ti=np.linspace(0, 1, num=C1.shape[0])
    X_t=skfda.FDataGrid(C1,ti )
    Y_t=skfda.FDataGrid(C1,ti )
    
    basis = skfda.representation.basis.FourierBasis(n_basis=M)
    
    X_basis = X_t.to_basis(basis)
    Y_basis = Y_t.to_basis(basis)
    
    A_1=X_basis.coefficients
    A_2=Y_basis.coefficients
    
    A=np.concatenate([A_1, A_2], axis=0)
    return A 


def Rot(theta): 
    return np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))
    
M=11

X_0=np.load('curves/Butterfly-1_0.npy')
Y_0=np.load('curves/Butterfly-1_1.npy')

if False : 
    X_trans=Rot(np.pi) @ np.concatenate([[X_0], [Y_0] ], axis=0)
    
    X_1=X_trans[0, :]
    Y_1=X_trans[1, :]

X_1=np.load('curves/Butterfly-10_0.npy')
Y_1=np.load('curves/Butterfly-10_1.npy')

B_0=get_basis(X_0, Y_0, M=21)
B_1=get_basis(X_1, Y_1, M=21)


#remove rho et translation
A_0=B_0[:, 1:]/np.linalg.norm(B_0[:, 1:])
A_1=B_1[:, 1:]/np.linalg.norm(B_1[:, 1:])


S=A_0 @ np.transpose(A_1)

SVD=np.linalg.svd(S)

O=SVD[0] @ SVD[2]
if np.linalg.det(O) < 0: 
    I_sign=np.identity(2)
    I_sign[1, 1]=-1
    O=SVD[0]@I_sign @ SVD[2]

Z= O@ np.concatenate([[X_1], [Y_1] ], axis=0)

##Z \sim X_0, Y_0

plt.plot(Z[0, :], Z[1, :])
plt.plot(X_0, Y_0)

plt.plot(X_1, Y_1)


plt.plot(Z[0, :]-np.mean(Z[0, :]))

plt.plot(X_0-np.mean(X_0))
plt.plot(X_1-np.mean(X_1))