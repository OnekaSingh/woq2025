import numpy as np

def average_weight(x, z) -> float:

    N = x.shape[0]

    average_weight = np.bitwise_or(x, z).sum() / N

    return average_weight