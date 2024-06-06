import numpy as np


def check_degeneracy(
    x_basis: np.array,
):
    return np.any(x_basis == 0)
