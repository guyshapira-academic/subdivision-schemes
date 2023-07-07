import numpy as np
from numpy.typing import NDArray

from scipy import linalg

import matplotlib.pyplot as plt


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


def spd_circle(n: int) -> NDArray:
    """
    Generates a circle of SPD matrices.

    Parameters:
        n: Number of matrices to generate.

    Returns:
        The generated SPD matrices. shape: (n, 3, 3)
        Coordinates of the generated matrices. shape: (n, 6)
    """
    # Generate the coordinates.
    coords = np.zeros((n, 6))
    coords[:, 0] = 0
    coords[:, 1] = np.sin(np.linspace(0, 2 * np.pi, n))
    coords[:, 2] = np.cos(np.linspace(0, 2 * np.pi, n))
    coords[:, 3] = 0
    coords[:, 4] = 0
    coords[:, 5] = 0

    # Random 6 x 6 matrix.
    random_matrix = np.random.rand(6, 6)
    coords = coords @ random_matrix

    # Convert to SPD matrices.
    matrices = coordinates_to_matrices(coords)

    return matrices, coords


def refinement_step(x: NDArray) -> NDArray:
    """Refinement step for the SPD matrices.

    Parameters:
        x: Input SPD matrices. shape: (N, 3, 3)

    Returns:
        The refined SPD matrices. shape: (N, 3, 3)
    """
    # Get the number of samples.
    N = x.shape[0]

    refined_data = np.zeros((2 * N, 3, 3))

    for i in range(N):
        # Even indices.
        refined_data[2 * i, :, :] = spd_average(
            x[[i - 2, i - 1, i]], np.array([1 / 8, 3 / 4, 1 / 8])
        )
        # Odd indices.
        refined_data[2 * i + 1, :, :] = spd_average(
            x[[i - 1, i]], np.array([1 / 2, 1 / 2])
        )

    return refined_data


def main():
    N = 10

    # Generate matrices
    matrices, coords = spd_circle(N)

    matrices_refined = matrices
    for i in range(2):
        matrices_refined = refinement_step(matrices_refined)

    ax = plt.axes(projection="3d")

    for i, color in enumerate(["red", "green", "blue"]):
        curve_coordinates = matrices[:, :, i]
        x = curve_coordinates[:, 0]
        y = curve_coordinates[:, 1]
        z = curve_coordinates[:, 2]
        ax.scatter3D(x, y, z, c=color, marker="X", label=f"original-{i}")
        ax.plot3D(x, y, z, color, linewidth=0.25, linestyle="--")

    for i, color in enumerate(["red", "green", "blue"]):
        curve_coordinates = matrices_refined[:, :, i]
        x = curve_coordinates[:, 0]
        y = curve_coordinates[:, 1]
        z = curve_coordinates[:, 2]
        ax.scatter3D(x, y, z, c=color, marker=".", label=f"refined-{i}")
        ax.plot3D(x, y, z, c=color, linewidth=1)
    ax.legend()
    plt.savefig("spd.png")
    plt.show()


if __name__ == "__main__":
    main()
