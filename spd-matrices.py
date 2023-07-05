from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from scipy import linalg


def matrix_log(A: NDArray) -> NDArray:
    """Matrix logarithm.

    Args:
        A: Input matrix. shape: (d, d) where d is the dimension of the data.

    Returns:
        The matrix logarithm of A. shape: (d, d) where d is the dimension of
            the data.
    """
    log = np.zeros_like(A)

    for i in range(A.shape[0]):
        log[i, :, :] = linalg.logm(A[i, :, :])

    return log


def coordinates_to_matrices(coords: NDArray) -> NDArray:
    """Converts 6D coordinates to SPD matrices.

    Parameters:
        coords: Input coordinates. shape: (N, 3) where N is the number of
            samples.
    """

    # Get the number of samples.
    N = coords.shape[0]

    # Create symmetric matrices.
    matrices = np.zeros((N, 3, 3))

    # Fill the matrices.
    matrices[:, 0, 0] = coords[:, 0]
    matrices[:, 1, 1] = coords[:, 1]
    matrices[:, 2, 2] = coords[:, 2]
    matrices[:, 0, 1] = coords[:, 3]
    matrices[:, 1, 0] = coords[:, 3]
    matrices[:, 0, 2] = coords[:, 4]
    matrices[:, 2, 0] = coords[:, 4]
    matrices[:, 1, 2] = coords[:, 5]
    matrices[:, 2, 1] = coords[:, 5]

    # Calculate SPD by matrix exponential.
    matrices = linalg.expm(matrices)

    return matrices


def matrices_to_coordinates(matrices: NDArray) -> NDArray:
    """Converts SPD matrices to 6D coordinates.

    Parameters:
        matrices: Input matrices. shape: (N, 3, 3) where N is the number of
            samples.
    """

    # Get the number of samples.
    N = matrices.shape[0]

    # Create coordinates.
    coords = np.zeros((N, 6))

    # Fill the coordinates.
    coords[:, 0] = matrices[:, 0, 0]
    coords[:, 1] = matrices[:, 1, 1]
    coords[:, 2] = matrices[:, 2, 2]
    coords[:, 3] = matrices[:, 0, 1]
    coords[:, 4] = matrices[:, 0, 2]
    coords[:, 5] = matrices[:, 1, 2]

    # Calculate SPD by matrix exponential.
    coords = matrix_log(coords)

    return coords


def binary_spd_average(matrices: NDArray, t: float) -> NDArray:
    """Binary SPD average.

    Args:
        matrices (NDArray): Input SPD matrices. shape: (2, 3, 3)
        t: Interpolation parameter.

    Returns:
        The binary SPD average of a and b. shape: (3, 3)
    """
    log_mean = (1 - t) * linalg.logm(matrices[0]) + t * linalg.logm(matrices[1])
    return linalg.expm(log_mean)


def spd_average(matrices: NDArray, weights: NDArray) -> NDArray:
    """Weighted SPD average.

    Args:
        matrices: Input SPD matrices. shape: (N, 3, 3)
        weights: Weights. shape: (N,)

    Returns:
        The weighted SPD mean of the input matrices. shape: (3, 3)
    """
    # Get the number of samples.
    N = len(matrices)
    weights = weights / np.sum(weights)

    # Calculate recursively using the binary spd average.
    if N == 2:
        return binary_spd_average(matrices, weights[1])
    else:
        term_1 = spd_average(matrices[:-1], weights[:-1])
        term_2 = matrices[-1]
        return binary_spd_average(np.array([term_1, term_2]), weights[-1])


if __name__ == "__main__":
    # Test the SPD average.

    # Create multiple SPD matrices.
    matrices = np.random.randn(10, 3, 3)
    matrices = np.matmul(matrices, np.transpose(matrices, (0, 2, 1)))
    matrices = linalg.expm(matrices)

    # Calculate the average.
    weights = np.random.rand(10)
    average = spd_average(matrices, weights)

    # Check if the average is SPD.
    print(np.all(linalg.eigvals(average) > 0))

    print(average)

