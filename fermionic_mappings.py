from typing import Union
import numpy as np
from qiskit_nature.second_q.mappers.bravyi_kitaev_mapper import BravyiKitaevMapper
from qiskit.quantum_info import PauliList
from itertools import chain

def jw_majoranas(N: int) -> Union[np.ndarray, np.ndarray]:

    x = np.zeros(shape=(2*N, N), dtype=bool)
    z = np.zeros(shape=(2*N, N), dtype=bool)

    for n in range(N):

        x[n, n] = x[n+N, n] = True
        z[n+N, n] = True

        z[n, :n] = True
        z[n+N, :n] = True

    return x, z

def bk_majoranas(N: int) -> Union[np.ndarray, np.ndarray]:

    pauli_table = BravyiKitaevMapper.pauli_table(N)

    pauli_table = list(chain.from_iterable(pauli_table))
    paulis = PauliList(pauli_table)

    x, z = paulis.x, paulis.z

    return x, z, paulis

