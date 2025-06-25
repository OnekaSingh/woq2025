import numpy as np
from electronic_hamiltonian import quadratic_terms_indices, quartic_terms_indices, quadratic_terms, quartic_terms
from qiskit_ibm_runtime.fake_provider import FakeMelbourneV2
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime.fake_provider import FakeGeneva
from qiskit_ibm_runtime.fake_provider import FakeAlgiers
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit.transpiler import CouplingMap
import networkx as nx
import copy, math, random
from numba import njit

@njit()
def weight(x, z) -> float:

    weight = np.bitwise_or(x, z).sum()

    return weight

@njit()
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



def compute_cost_pauli_string(x, z, coupling_map=None):
    if coupling_map is None:
        coupling_map = CouplingMap(FakeTorino().coupling_map)
    
    # Get qubit indices involved in this Pauli string
    involved_qubits = [i for i, (xi, zi) in enumerate(zip(x, z)) if xi or zi]
    # print(involved_qubits)
    
    if len(involved_qubits) <= 1:
        return 0  # Single-qubit ops are free

    # Build full graph of pairwise distances
    G = nx.Graph()
    for i in range(len(involved_qubits)):
        for j in range(i+1, len(involved_qubits)):
            q1, q2 = involved_qubits[i], involved_qubits[j]
            dist = coupling_map.distance(q1, q2)
            G.add_edge(q1, q2, weight=dist)

    # Minimum Spanning Tree to connect all involved qubits
    mst = nx.minimum_spanning_tree(G)
    cost = sum([d['weight'] for u, v, d in mst.edges(data=True)])

    return cost

# cost = compute_cost_pauli_string([True, True, False, False, False, False, False, False, False, False],
#                                  [True, False, False, False, False, False, False, False, False, False],
#                                  CouplingMap(FakeBrisbane().coupling_map))


def compute_cost_pauli_string1(x, z, coupling_map=None, logical_to_physical=None):
    if coupling_map is None:
        coupling_map = CouplingMap(FakeTorino().coupling_map)

    # Get logical qubit indices where operator acts
    involved_logical = [i for i, (xi, zi) in enumerate(zip(x, z)) if xi or zi]
    if len(involved_logical) <= 1:
        return 0  # No two-qubit interactions needed

    # Apply mapping if provided, else use identity
    if logical_to_physical is None:
        logical_to_physical = {i: i for i in involved_logical}

    try:
        involved_physical = [logical_to_physical[q] for q in involved_logical]
    except KeyError as e:
        raise ValueError(f"Missing mapping for logical qubit {e.args[0]}")

    # Build full weighted graph of distances
    G = nx.Graph()
    for i in range(len(involved_physical)):
        for j in range(i+1, len(involved_physical)):
            q1, q2 = involved_physical[i], involved_physical[j]
            dist = coupling_map.distance(q1, q2)
            G.add_edge(q1, q2, weight=dist)

    # MST gives cost of best connectivity
    mst = nx.minimum_spanning_tree(G)
    cost = sum(d['weight'] for _, _, d in mst.edges(data=True))

    return cost




def simulated_annealing_mapping(x, z, coupling_map, max_iter=100, initial_temp=1000.0, cooling_rate=0.95):
    # Identify involved logical qubits
    involved_logical = [i for i, (xi, zi) in enumerate(zip(x, z)) if xi or zi]
    num_logical = len(involved_logical)

    # Get available physical qubits (you can customize this)
    available_physical = list(set(range(coupling_map.size())))

    if len(available_physical) < num_logical:
        raise ValueError("Not enough physical qubits for the logical operators.")

    # Initialize: pick a random mapping
    current_mapping = dict(zip(involved_logical, random.sample(available_physical, num_logical)))
    current_cost = compute_cost_pauli_string1(x, z, coupling_map, current_mapping)
    best_mapping = current_mapping
    best_cost = current_cost

    temperature = initial_temp

    for iteration in range(max_iter):
        # Propose new mapping: swap two physical assignments
        new_mapping = copy.deepcopy(current_mapping)
        q1, q2 = random.sample(involved_logical, 2)
        new_mapping[q1], new_mapping[q2] = new_mapping[q2], new_mapping[q1]

        new_cost = compute_cost_pauli_string1(x, z, coupling_map, new_mapping)

        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_mapping = new_mapping
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_mapping = new_mapping

        temperature *= cooling_rate  # Cool down

        if temperature < 1e-3:
            break

    return best_mapping, best_cost

def generate_random_mapping(num_logical_qubits, coupling_map: CouplingMap):
    physical_qubits = list(range(coupling_map.size()))

    if num_logical_qubits > len(physical_qubits):
        raise ValueError("Not enough physical qubits to map all logical qubits.")

    selected_physical = random.sample(physical_qubits, num_logical_qubits)
    mapping = {logical: physical for logical, physical in enumerate(selected_physical)}
    return mapping

def compute_cost_pauliString_circuitCoupling(x, y, map = CouplingMap(FakeBrisbane().coupling_map)):

    """
    Computes the cost of a Pauli string circuit given a coupling map.

    Args:
        pauliString (binary string): The Pauli string to be evaluated.
        map (CouplingMap): The coupling map of the quantum device.

    Returns:
        int: The cost of the circuit.
    """
    cost_total = 0
    mapping = generate_random_mapping(num_logical_qubits=len(x[0]), coupling_map=map)
    for j in range(len(x)):
        # print(j)
        
        # cost = compute_cost_pauli_string
        # map = {}
        #generate a random mapping with the number of qubits given by the map
        

        # mapping = {0: 1, 1: 2, 2: 3}
        cost = compute_cost_pauli_string1(x[j], y[j], map, mapping)
        # _, cost = simulated_annealing_mapping(x[j], y[j], map)
        cost_total += cost

    return cost_total
