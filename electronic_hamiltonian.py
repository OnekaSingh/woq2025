from itertools import combinations
import numpy as np
from numba import njit

def quadratic_terms_indices(N):
    return list(combinations(range(N), 2))

def quartic_terms_indices(N):
    return list(combinations(range(N), 4))

@njit
def quadratic_terms_indices_numba(N):
    count = N * (N - 1) // 2 
    result = np.zeros((count, 2), dtype=np.int64)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            result[idx, 0] = i
            result[idx, 1] = j
            
            idx += 1

    return result

@njit
def quartic_terms_indices_numba(N):
    count = N * (N - 1) * (N - 2) * (N - 3) // 24  # C(N,4)
    result = np.zeros((count, 4), dtype=np.int64)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    result[idx, 0] = i
                    result[idx, 1] = j
                    result[idx, 2] = k
                    result[idx, 3] = l

                    idx += 1

    return result

@njit
def new_quadratic_terms_lookup(N: int):

    lookup_table = np.zeros(shape=(N, N-1), dtype=np.int16)

    for i in range(N):

        idx = 0
        
        for j in range(N):
            
            if i == j:
                idx -= 1
            else:
                lookup_table[i, idx] = j

                idx += 1

    return lookup_table

@njit
def quadratic_terms(x, z, N):

    # indices = quadratic_terms_indices(N)
    indices = quadratic_terms_indices_numba(N)
    num_terms = indices.shape[0]

    # print(indices)

    x_terms = np.zeros(shape=(num_terms, N), dtype=np.uint8)
    z_terms = np.zeros(shape=(num_terms, N), dtype=np.uint8)

    for row, (i, j) in enumerate(indices):

        x_terms[row] = np.bitwise_xor(x[i], x[j])
        z_terms[row] = np.bitwise_xor(z[i], z[j])

    return x_terms, z_terms

@njit
def quartic_terms(x, z, N):

    # indices = quartic_terms_indices(N)
    indices = quartic_terms_indices_numba(N)
    num_terms = indices.shape[0]# len(indices)

    x_terms = np.zeros(shape=(num_terms, N), dtype=np.uint8)
    z_terms = np.zeros(shape=(num_terms, N), dtype=np.uint8)

    for row, (i, j, k, l) in enumerate(indices):

        x_terms[row] = np.bitwise_xor(np.bitwise_xor(x[i], x[j]), np.bitwise_xor(x[k], x[l]))
        z_terms[row] = np.bitwise_xor(np.bitwise_xor(z[i], z[j]), np.bitwise_xor(z[k], z[l]))

    return x_terms, z_terms

