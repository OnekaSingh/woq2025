import numpy as np
from numba import njit

@njit(fastmath=True)
def spread_node(n: int, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    
    N = x.shape[0]

    x_n, z_n = x[n], z[n]

    for i in range(N):

        if i != n:

            x[i] = np.bitwise_xor(x[i], x_n)
            z[i] = np.bitwise_xor(z[i], z_n)

    return x, z

@njit(fastmath=True)
def spread_node_slice(n: int, x: np.ndarray, z: np.ndarray):
    x_n = x[n].copy()   # preserve the pivot‐row
    z_n = z[n].copy()

    # XOR all rows < n with x_n/z_n
    x[:n]   ^= x_n
    z[:n]   ^= z_n
    # XOR all rows > n
    x[n+1:] ^= x_n
    z[n+1:] ^= z_n

    # row n is unchanged
    return x, z



def binary_matmul_xor(A, B):
    """
    Perform binary matrix multiplication using bitwise XOR as addition
    and bitwise AND as multiplication.
    Assumes A and B are binary arrays of shape (m, n) and (n, p) respectively.
    """
    A = A.astype(np.uint8)
    B = B.astype(np.uint8)
    
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "Incompatible shapes for matrix multiplication"
    
    result = np.zeros((m, p), dtype=np.uint8)
    for i in range(m):
        for j in range(p):
            anded = A[i, :] & B[:, j]
            result[i, j] = np.bitwise_xor.reduce(anded)
    
    return result

def anticommutation_matrix(x, z) -> np.ndarray:
    """Compute the anticommutation graph of a stabilizer tableau.

    Args:
        x (_type_): X
        z (_type_): Z

    Returns:
        np.ndarray: Adjacency matrix of the anticommutation graph
    """

    return np.bitwise_xor(binary_matmul_xor(x, z.T), binary_matmul_xor(z, x.T))