import numpy as np
from copy import deepcopy
import random

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

def reconstruct(mps_in):
    state_vec = []

    mps = deepcopy(mps_in)

    N = len(mps)

    # this converts to a different index order
    # where the physical index is the outermost index and the bond index is contained
    # also puts into canonical form we are used to seeing
    mps[-1] = np.swapaxes(mps[-1], 0, 1)
    for i in range(len(mps)-1):
        mps[i][0] = np.swapaxes(mps[i][0], 0, 1)

    for i in range(2**N):
        ind = [int(bit) for bit in format(i, '0{}b'.format(N))]
        prod = [[row[ind[-1]]] for row in mps[-1]]
        for j in range(len(mps)-2, -1, -1):
            tensors = mps[j]
            prod = tensors[1] @ prod
            prod = tensors[0][ind[j]] @ prod
        state_vec.append(prod[0])
    return np.abs(state_vec) / np.linalg.norm(state_vec)

def contract(mps, N):
    contraction = mps[-1]


def mps_3(state_v):
    N = 3
    f = np.reshape(state_v, (2, 2**(N-1)))
    gamma1, lambda1, V1 = np.linalg.svd(f, full_matrices=False)
    #truncate then normalize
    lambda1 = lambda1/np.linalg.norm(lambda1)
    # reshape V1
    V1 = np.reshape(V1, (2**2, 2**(N-2)))
    gamma2, lambda2, V2 = np.linalg.svd(V1, full_matrices=False)
    gamma2 = np.reshape(gamma2, (2, 2, 2))
    lambda2 = lambda2 / np.linalg.norm(lambda2)
    gamma3, lambda3, V3 = np.linalg.svd(V2, full_matrices=False)

    lambda1 = np.diag(lambda1)
    lambda2 = np.diag(lambda2)
    lambda3 = np.diag(lambda3)

    mps = [[gamma1, lambda1], [gamma2, lambda2], gamma3]
    return mps

def mps(state_v, chi=1000000):
    mps_v = []
    N = int(np.log2(len(state_v)))
    right = np.reshape(state_v, (2, 2**(N-1)))
    for i in range(N):
        if i == N-1:
            mps_v.append(right)
            continue
        gamma, S, right = np.linalg.svd(right, full_matrices=False)
        # left and right most gammas are our MPS caps, only have one bond index
        if i > 0 and i < N-1:
            if chi < len(S) and chi >= 2:
                gamma = gamma[:, :chi]
                S = S[:chi]
                right = right[:chi, :]

            gamma = np.reshape(gamma, (int(gamma.shape[0]/2), 2, gamma.shape[1]))

        if i+2 < N:
            right = np.reshape(right, (int(right.shape[0]*2), int(right.shape[1]/2)))

        lambd = np.diag(S/np.linalg.norm(S))
        mps_v.append([gamma, lambd])
    return mps_v

# not really a haar state lol
def haar_state(N):
    state = [random.random() for _ in range(2**N)]
    return state / np.linalg.norm(state)

def test_mps(N, trials, chi):
    avg = 0
    for i in range(trials):
        state = haar_state(N)
        # if random.random() < 0.05:
        #     print(state)
        #     print(reconstruct(mps(state)))
        # **(int(N/2)-6))
        m = mps(state, chi=chi)
        avg += np.dot(state, reconstruct(m))
    return avg / trials



if __name__ == "__main__":
    # state_v = [1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)] # GHZ
    # state_v = [0, 0, 1/np.sqrt(2), 0, 0, 1/np.sqrt(2), 0, 0] # anti-ferromagnetic GHZ
    # state_v = [0, 1/np.sqrt(3), 1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 0, 0] # W state
    # state_v = [1, 0, 1, 1, 1, 0, 0, 1]
    # state_v = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    # state_v = [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # state_v = state_v / np.linalg.norm(state_v)
    # mps_v = mps(state_v)
    # print(mps_v)
    # for i in mps_v:
    #     print(i)
    # rec = np.abs(reconstruct(mps_v))
    # print(rec)
    # print(np.linalg.norm(rec))
    # print(np.linalg.norm(reconstruct(mps_v)))
    # rec = rec / np.linalg.norm(rec)
    # print(np.dot(rec, state_v))
    # for i in range(5, 20):
    #     similarity = test_mps(N=i, trials=1, chi=2)
    #     print("%d qubit fidelity: %f" % (i, similarity))
    data = []
    for N in [2, 4, 6, 8, 10, 12]:
        line = []
        for i in range(2, 50):
            # similarity = test_mps(N=12, trials=1, chi=i)
            # print("%d bond dimension fidelity: %f" % (i, similarity))
            line.append(test_mps(N=12, trials=1, chi=i))
        data.append(line)
    
