import numpy as np
from electronic_hamiltonian import quadratic_terms_indices, quartic_terms_indices, quadratic_terms, quartic_terms
from numba import njit

@njit()
def weight(x, z) -> float:

    weight = np.bitwise_or(x, z).sum()

    return weight

@njit
def quadratic_term_mean_weight(x: np.ndarray, z: np.ndarray) -> float:

    N = x.shape[1]

    x_terms, z_terms = quadratic_terms(x, z, N)
    num_terms = x_terms.shape[0]

    total_weight = weight(x=x_terms, z=z_terms)

    return total_weight/num_terms


@njit
def quartic_term_mean_weight(x: np.ndarray, z: np.ndarray) -> float:

    N = x.shape[1]

    x_terms, z_terms = quartic_terms(x, z, N)
    num_terms = x_terms.shape[0]

    total_weight = weight(x=x_terms, z=z_terms)

    return total_weight/num_terms


#def compute_cost_pauliString_circuitCoupling(x, y, map = CouplingMap(FakeMelbourneV2().coupling_map)):

    """
    Computes the cost of a Pauli string circuit given a coupling map.

    Args:
        pauliString (binary string): The Pauli string to be evaluated.
        map (CouplingMap): The coupling map of the quantum device.

    Returns:
        int: The cost of the circuit.
    """
    cost_total = 0
    for j in range(len(x)):
        indexes = []
        for (i,el) in enumerate(x[j]):
            # print(el)
        #     temp = []
            # print(x[j][i], y[j][i])
            temp = [x[j][i], y[j][i]]
            # print(temp)
            if True in temp:
                indexes.append(i)

        # for j in x[i]:
            # print(j)
        # if True in x[i] or True in y[i]:

            # indexes.append(i)


        cost = 0
        for i in range(len(indexes) - 1):
            cur_cost = map.distance(indexes[i], indexes[i+1])
            # print(cur_cost)
            cost += cur_cost
        if cost != len(indexes) - 1:
            # return cost
            print('da')
            cost_total += 10*cost
        else:
            # return 0
            cost_total += cost

    return cost_total
#