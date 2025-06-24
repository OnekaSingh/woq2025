from itertools import combinations
import numpy as np

def quadratic_terms_indices(N):
    return list(combinations(range(N), 2))

def quartic_terms_indices(N):
    return list(combinations(range(N), 4))

def quadratic_terms(x, z, N):

    indices = quadratic_terms_indices(N)
    num_terms = len(indices)

    # print(indices)

    x_terms = np.zeros(shape=(num_terms, N), dtype=bool)
    z_terms = np.zeros(shape=(num_terms, N), dtype=bool)

    for row, (i, j) in enumerate(indices):

        x_terms[row] = np.bitwise_xor(x[i], x[j])
        z_terms[row] = np.bitwise_xor(z[i], z[j])

    return x_terms, z_terms