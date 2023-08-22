import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigs
from copy import deepcopy
import random
import numba as nb
import time
import warnings
warnings.filterwarnings("ignore")

V_nn = 2*np.pi*60*(10**6)
V_nnn = 2*np.pi*2.3*(10**6)
C6 = 2 * np.pi * 275 * (10**9)
lattice_spacing = 6

def tensor(A, B):
    m = len(A)*len(B)
    n = len(A[0])*len(B[0])

    prod = []

    for i in range(m):
        prod.append([])
        for j in range(n):
            ax = int(i / len(B))
            ay = int(j / len(B[0]))
            bx = int(i % len(B))
            by = int(j % len(B[0]))
            prod[i].append(A[ax][ay]*B[bx][by])

    return np.array(prod)

def tensor_list(ops):
    prod = ops[-1]
    for i in range(len(ops)-2, -1, -1):
        prod = tensor(ops[i], prod)
    return np.array(prod)

def pauli(pos, dir, N):
    factors = [[[1, 0], [0, 1]]]*N
    if dir == 0 or dir == 'x':
        factors[pos] = [[0, 1], [1, 0]]
    elif dir == 1 or dir == 'y': # y
        factors[pos] = [[0, -1j], [1j, 0]]
    elif dir == 2 or dir == 'z': # z
        factors[pos] = [[1, 0], [0, -1]]
    elif dir == 'n':
        factors[pos] = [[0, 0], [0, 1]]

    return tensor_list(factors)

def interaction_strength(a, b):
    return C6 / (abs(a - b) * lattice_spacing)**6

def hamiltonian(N, rabi_f, delta):
    H = -0.5 * rabi_f * sum([pauli(i, 'x', N) for i in range(N)])
    H -= sum([delta * pauli(i, 'n', N) for i in range(N)])

    interaction_term = sum([sum([interaction_strength(i, j) * pauli(i, 'n', N) @ pauli(j, 'n', N) for j in range(i)]) for i in range(N)])
    H += interaction_term
    return H

def v_entropy(rho_eigenvalues):
    """Compute the von Neumann entropy of a density matrix given its eigenvalues."""
    return -np.sum(rho_eigenvalues * np.log2(rho_eigenvalues + 1e-10))

def bipartite_entropy(state_vector):
    """Compute the bipartite von Neumann entropy of an N=3 system."""
    N = np.log2(len(state_vector))
    # Reshape the state vector into a 2x4 matrix
    psi_matrix = state_vector.reshape(int(2**(int(N/2))), int(2**(N - int(N/2))))
    
    # Perform SVD
    U, singular_values, Vh = np.linalg.svd(psi_matrix, full_matrices=False)
    
    # Square the singular values to get the eigenvalues of the reduced density matrix
    rho_eigenvalues = singular_values**2

    # print(rho_eigenvalues)
    
    # Compute the entropy
    entropy = v_entropy(rho_eigenvalues)
    
    return entropy


def ED(rabi_f, detuning):
    N = 9
    eigenvalues, eigenvectors = eigs(hamiltonian(N, rabi_f, detuning))

    min_eigenvalue_index = np.argmin(eigenvalues)
    # print(eigenvalues)

    min_eigenvalue = eigenvalues[min_eigenvalue_index]
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    # print(min_eigenvector)
    return min_eigenvector

# 2*np.pi*6.4*(10**6)
# 2*np.pi*60*(10**6)
# 2*np.pi*2.3*(10**6)

resolution = 10
frequencies = np.linspace(1, 5, num=resolution)
ratios = np.linspace(0, 6, num=resolution)

grid_data = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        print(i * resolution + j)
        rabi_freq = C6/((lattice_spacing * frequencies[i])**6)
        grid_data[i][j] = bipartite_entropy(ED(rabi_freq, rabi_freq * ratios[j]))

# frequencies = sorted(frequencies)
# grid_data = reversed(grid_data)

X, Y = np.meshgrid(ratios, frequencies)

plt.figure(figsize=(8, 6))
plt.imshow(grid_data, extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)), origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Bipartite Entropy')

plt.xlabel('R_b/a')
plt.ylabel('Δ/Ω')

plt.show()