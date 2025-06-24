from typing import Union
import numpy as np

def jw_majoranas(N: int) -> Union[np.ndarray, np.ndarray]:

    x = np.zeros(shape=(2*N, N), dtype=bool)
    z = np.zeros(shape=(2*N, N), dtype=bool)

    for n in range(N):

        x[n, n] = x[n+N, n] = True
        z[n+N, n] = True

        z[n, :n] = True
        z[n+N, :n] = True

    return x, z

