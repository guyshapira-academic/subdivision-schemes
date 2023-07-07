import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt


def binary_sphere_distance(points: NDArray) -> NDArray:
    """Calculates the distance between pairs of two points on the sphere.

    Parameters:
        points: Input points. shape: (N, 2, 3).
    """
    # Calculate the distance.
    distance = np.arccos(np.sum(points[:, 0, :] * points[:, 1, :], axis=-1))
    return distance


def sphere_log(x: NDArray) -> NDArray:
    """Logarithm map on the sphere, w.r.t the point
        (1, 0, 0)

    Parameters:
        x: Input points. shape: (N, 3).
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    x0 = np.array([1, 0, 0]).reshape(1, -1)
    x0 = np.tile(x0, (x.shape[0], 1))


    # Calculate the distance from the point (1, 0, 0)
    distance = binary_sphere_distance(np.stack([x0, x], axis=1))

    # Calculate the logarithm.
    log = np.zeros((x.shape[0], 2))
    log[:, 0] = x[:, 1]
    log[:, 1] = x[:, 2]
    log /= np.linalg.norm(log, axis=-1, keepdims=True)
    log *= distance.reshape(-1, 1)

    return log


def sphere_exp(v: NDArray) -> NDArray:
    """Exponential map on the sphere, w.r.t the point
        (1, 0, 0)

    Parameters:
        v: Input vectors from the tangent space of (1, 0, 0). shape: (N, 3).
    """
    x0 = np.array([1, 0, 0]).reshape(1, -1)
    x0 = np.tile(x0, (v.shape[0], 1))

    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.concatenate([np.zeros((v.shape[0], 1)), v], axis=-1)

    return np.cos(v_norm) * x0 + np.sin(v_norm) * v / v_norm


def circle_projection(n: int) -> NDArray:
    """Creates a closed loop on the
    2D unit sphere.

    Parameters:
            n: Number of points to generate.
    """
    # Generate the coordinates.
    coords = np.zeros((n, 3))
    t = np.linspace(0, 1, n)
    phi = 0.9 * 0.5 * np.pi * np.cos(np.pi * t)
    theta = 0.5 * (0.5 * np.pi * np.sin(np.pi * t) + 0.2 * np.cos(4 * np.pi * t ** 2))
    coords[:, 0] = np.sin(theta) * np.cos(phi)
    coords[:, 1] = np.sin(theta) * np.sin(phi)
    coords[:, 2] = np.cos(theta)

    return coords


def binary_sphere_average(points: NDArray, t: float) -> NDArray:
    """Calculates the average of a set of pairs of points on the sphere.

    Parameters:
        points: Input points. shape: (2, 3).
        t: Interpolation parameter.
    """
    log_1 = sphere_log(points[0, :])
    log_2 = sphere_log(points[1, :])

    # Calculate the exponential.
    exp = sphere_exp((1 - t) * log_1 + t * log_2)

    return exp


def sphere_average(points: NDArray, weights: NDArray) -> NDArray:
    """Calculates the average of a set of points on the sphere.

    Parameters:
        points: Input points. shape: (N, 3).
        weights: Weights of the points. shape: (N,).
    """
    N = len(points)
    weights = weights / np.sum(weights)

    if N == 2:
        return binary_sphere_average(points, weights[1])
    else:
        term_1 = sphere_average(points[:-1], weights[:-1])
        if len(term_1.shape) == 1:
            term_1 = term_1.reshape(1, -1)
        term_2 = points[-1].reshape(1, -1)

        return binary_sphere_average(np.array([term_1, term_2]), weights[-1])


def refinement_step(x: NDArray) -> NDArray:
    """Refinement step for points on the sphere

    Parameters:
        x: Input points. shape: (N, 3)

    Returns:
        The refined points. shape: (N, 3)
    """
    # Get the number of samples.
    N = x.shape[0]

    refined_data = np.zeros((2 * N, 3))

    for i in range(N):
        # Even indices.
        refined_data[2 * i, :] = sphere_average(
            x[[i - 2, i - 1, i]], np.array([1 / 8, 3 / 4, 1 / 8])
        )
        # Odd indices.
        refined_data[2 * i + 1, :] = sphere_average(
            x[[i - 1, i]], np.array([1 / 2, 1 / 2])
        )

    return refined_data


def main():
    x = circle_projection(10)

    ax = plt.axes(projection="3d")

    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    x_ = np.cos(u) * np.sin(v)
    y_ = np.sin(u) * np.sin(v)
    z_ = np.cos(v)
    ax.plot_surface(x_, y_, z_, color="gray", alpha=0.2)

    x_refined = x
    for i in range(2):
        x_refined = refinement_step(x_refined)
    ax.scatter3D(x_refined[:, 0], x_refined[:, 1], x_refined[:, 2], marker=".", color="red", label="refined")
    ax.plot3D(x_refined[:, 0], x_refined[:, 1], x_refined[:, 2], color="red", linewidth=1)

    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], color="blue", marker="X", label="original")
    ax.plot3D(x[:, 0], x[:, 1], x[:, 2], color="blue", linewidth=0.25)

    ax.view_init(elev=35, azim=30, roll=0)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()