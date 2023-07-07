import numpy as np
from numpy.typing import NDArray
from scipy import fft

import matplotlib.pyplot as plt


def circular_convolve(x: NDArray, y: NDArray) -> NDArray:
    """Circular convolution of x and y.

    Args:
        x: Input array. shape: (N, d) or (N, ) where N is the number of
            samples and d is the dimension of the data.
        y: input array. shape: (N, d) or (N, ) where N is the number of
            samples and d is the dimension of the data.

    Returns:
        The circular convolution of x and y. shape: (N, d) or (N, ) where N
            is the number of samples and d is the dimension of the data.
    """
    return np.real(fft.ifft(fft.fft(x, axis=0) * fft.fft(y, axis=0), axis=0))


def even_kernel(N: int) -> NDArray:
    """Even-points kernel.

    Args:
        N: Number of samples.

    Returns:
        Even kernel. shape: (N, 1) where d is the dimension of the data.
    """
    kernel = np.zeros((N, 1))
    kernel[-2] = 1 / 8
    kernel[-1] = 3 / 4
    kernel[0] = 1 / 8

    return kernel


def odd_kernel(N: int) -> NDArray:
    """Even-points kernel.

    Args:
        N: Number of samples.

    Returns:
        Even kernel. shape: (N, 1) where d is the dimension of the data.
    """
    kernel = np.zeros((N, 1))
    kernel[-1] = 1 / 2
    kernel[0] = 1 / 2
    return kernel


def refine_step(points: NDArray) -> NDArray:
    """Refine step.

    Args:
        points: Input points. shape: (N, d) where N is the number of samples
            and d is the dimension of the data.

    Returns:
        Refine points. shape: (2N, d) where N is the number of samples and d
            is the dimension of the data.
    """
    N, d = points.shape
    even_points = circular_convolve(points, even_kernel(N))
    odd_points = circular_convolve(points, odd_kernel(N))
    new_points = np.zeros((2 * N, d))
    new_points[0::2] = even_points
    odd_points = np.roll(odd_points, -1, axis=0)
    new_points[1::2] = odd_points
    return new_points


def main():
    import matplotlib.pyplot as plt

    points = np.array([[-1, 1], [-0.5, 0.75], [0.5, 0.75], [1, 1], [0, -1]])
    # t = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    # points = np.array([np.cos(t), np.sin(t)]).T

    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y)

    new_points = points
    points_ = np.concatenate([points, points[[0], :]], axis=0)
    plt.scatter(points_[:, 0], points_[:, 1], c="r", marker="X", label="original")
    plt.plot(points_[:, 0], points_[:, 1], c="r", linewidth=0.25, linestyle="--")

    for _ in range(1):
        new_points = refine_step(new_points).copy()
    plt.scatter(new_points[:, 0], new_points[:, 1], c="b", marker=".", label="refined-1")
    plt.plot(new_points[:, 0], new_points[:, 1], c="b", linewidth=1, linestyle="-")

    for _ in range(10):
        new_points = refine_step(new_points).copy()
    plt.plot(new_points[:, 0], new_points[:, 1], c="black", linewidth=1, linestyle="-", label="refined-10")
    plt.legend()
    plt.savefig("cubic-spline.png")
    plt.show()


if __name__ == "__main__":
    main()
