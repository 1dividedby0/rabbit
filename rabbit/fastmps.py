import matplotlib.pyplot as plt
import numpy as np
import scipy
from copy import deepcopy
import random
import numba as nb
import time

@nb.njit('complex128[:](int_)', parallel=True)
def haar_state(N):
    real_part = np.empty(2**N)
    imag_part = np.empty(2**N)

    # Parallel loop
    for i in nb.prange(2**N):
        real_part[i] = np.random.normal()
        imag_part[i] = np.random.normal()

    state = real_part + 1j * imag_part
    return state

def binary_mps(state_v, chi=1e50):
    N = int(np.log2(len(state_v)))
    dims = (int(N/2), int(N/2))
    if N%2==1:
        dims = (int(N/2), int(N/2)+1)
    start = np.reshape(state_v, (2**dims[0], 2**dims[1]))
    left, S, right = np.linalg.svd(start, full_matrices=False)
    mps_v = rec(left, (0, 0), True, 1, dims[0], chi=chi) + [np.diag(norm(S))] + rec(right, (0,0), False, 1, dims[1], chi=chi)
    mps_f = []
    for i in range(0, len(mps_v)-1, 2):
        mps_f.append([mps_v[i], mps_v[i+1]])
    mps_f.append(mps_v[-1])
    return mps_f

def rec(state_v, parent_shape, left, depth, N, chi=1e50):
    if (state_v.shape[0] == parent_shape[0] and state_v.shape[1] == parent_shape[1]) or (state_v.shape[0] == 2 and state_v.shape[1] == 2):
        if left == False and state_v.shape[1] > 2:
            state_v = np.reshape(state_v, (state_v.shape[0], 2, int(state_v.shape[1]/2)))
        elif left == True and state_v.shape[0] > 2:
            state_v = np.reshape(state_v, (int(state_v.shape[0]/2), 2, state_v.shape[1]))
        return [state_v]

    new_ = 0
    if left == True:
        new_ = np.reshape(state_v, (int(state_v.shape[0]/2), state_v.shape[1]*2))
    elif left == False:
        new_ = np.reshape(state_v, (state_v.shape[0]*2, int(state_v.shape[1]/2)))

    U, S, V = np.linalg.svd(new_, full_matrices=False)
    dims = [int(N/2), int(N/2)]
    if N %2 == 1:
        dims[1]+=1
    return rec(U, new_.shape, True, depth+1, dims[0], chi) + [np.diag(norm(S))] + rec(V, new_.shape, False, depth+1, dims[1], chi)


    