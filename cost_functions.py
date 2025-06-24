import numpy as np
from electronic_hamiltonian import quadratic_terms_indices, quartic_terms_indices, quadratic_terms

def weight(x, z) -> float:

    weight = np.bitwise_or(x, z).sum()

    return weight

def quadratic_term_mean_weight(x: np.ndarray, z: np.ndarray) -> float:

    N = x.shape[1]

    x_terms, z_terms = quadratic_terms(x, z, N)
    num_terms = x_terms.shape[0]

    total_weight = weight(x=x_terms, z=z_terms)

    return total_weight/num_terms
