""" Functions for generating and drawing random lines and curves. """

from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline
from skimage.morphology import dilation


def get_random_point_on_side(
    side: Literal["left", "right", "top", "bottom"], height: int, width: int
) -> tuple[int, int]:
    """Generate a random point on a given side of a rectangle.

    Parameters
    ----------
    side
        The side of the rectangle to generate the point on.
    height
        The height of the rectangle.
    width
        The width of the rectangle.

    Returns
    -------
    A tuple of two integers representing row and column coordinates of the point.
    """
    if side == "left":
        return (np.random.randint(0, height), 0)
    elif side == "right":
        return (np.random.randint(0, height), width - 1)
    elif side == "top":
        return (0, np.random.randint(0, width))
    elif side == "bottom":
        return (height - 1, np.random.randint(0, width))


def get_random_points_on_different_sides(
    height: int, width: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generate two random points on different sides of a rectangle.

    Parameters
    ----------
    height
        The height of the rectangle.
    width
        The width of the rectangle.

    Returns
    -------
    A tuple of two integers representing row and column coordinates of the point.
    """
    sides = ["left", "right", "top", "bottom"]
    side1, side2 = np.random.choice(sides, 2, replace=False)

    point1 = get_random_point_on_side(side1, height, width)
    point2 = get_random_point_on_side(side2, height, width)

    return point1, point2


def generate_random_line(
    height: int, width: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generate two random points on rectangle boundary.

    Parameters
    ----------
    height
        The height of the rectangle.
    width
        The width of the rectangle.

    Returns
    -------
    A tuple of two integers representing row and column coordinates of the point.
    """
    return get_random_points_on_different_sides(height, width)


def sample_points_on_line(
    p1: tuple[int, int], p2: tuple[int, int], num_samples: int
) -> np.ndarray:
    """Sample points on the line segment defined by p1 and p2.

    Parameters
    ----------
    p1
        The first point of the line segment represented as (row, col).
    p2
        The second point of the line segment represented as (row, col).
    num_samples
        The number of samples to generate.

    Returns
    -------
    A numpy array of shape (num_samples, 2) containing the sampled points.
    """
    # Sample more points, so they are not uniformly distributed
    num_generated_samples = num_samples * 2
    x_values = np.linspace(p1[0], p2[0], num_generated_samples)
    y_values = np.linspace(p1[1], p2[1], num_generated_samples)

    # Randomly choose indices to sample
    idxs = (
        [0]
        + list(
            np.random.choice(np.arange(1, num_generated_samples - 1), num_samples - 2)
        )
        + [num_generated_samples - 1]
    )
    idxs = np.sort(idxs)

    x_values = x_values[idxs]
    y_values = y_values[idxs]
    return np.column_stack((x_values, y_values))


def perturbate_points(
    points: np.ndarray, perturbation_strength: float, height: int, width: int
) -> np.ndarray:
    """Perturbate points on a line by adding random noise.

    Parameters
    ----------
    points
        The points to perturbate, represented as a numpy array of shape (num_points, 2)
        where each row contains the row and column coordinates of a point.
    perturbation_strength
        The strength of the perturbation.
    height
        The height of the image.
    width
        The width of the image.

    Returns
    -------
    A numpy array of shape (num_points, 2) containing the perturbed points.
    """
    perturbed_points = points + np.random.normal(
        scale=perturbation_strength, size=points.shape
    )

    # Don't change the first and last point
    perturbed_points[0] = points[0]
    perturbed_points[-1] = points[-1]

    perturbed_points[:, 0] = np.clip(perturbed_points[:, 0], 0, height)
    perturbed_points[:, 1] = np.clip(perturbed_points[:, 1], 0, width)
    return perturbed_points


def interpolate_curve(points: np.ndarray, n: int) -> np.ndarray:
    """Interpolate a smooth curve passing through given points

    Parameters
    ----------
    points
        The points to interpolate, represented as a numpy array of shape (num_points, 2)
        where each row contains the row and column coordinates of a point.
    n
        The number of points to interpolate between the given points.

    Returns
    -------
    A numpy array of shape (n, 2) containing the interpolated points.
    """

    t = np.arange(len(points))
    cs = CubicSpline(t, points)
    interpolated_curve = cs(np.linspace(0, len(points) - 1, n))
    return interpolated_curve


def draw_curve(
    curve: np.ndarray, width: int, height: int, thickness: int
) -> np.ndarray:
    """Return an image with a curve drawn on it.

    Parameters
    ----------
    curve
        The curve to draw, represented as a numpy array of shape (num_points, 2)
        where each row contains the row and column coordinates of a point.
    width
        The width of the image.
    height
        The height of the image.
    thickness
        The thickness of the curve.

    Returns
    -------
    A numpy array of shape (height, width) containing the image with the curve drawn
    on it. The background has value 0 and the curve has value 1.

    """
    img = np.zeros((height, width), dtype=np.uint8)
    curve = curve.astype(int)
    curve = curve[(curve[:, 0] < height) & (curve[:, 1] < width)]
    img[(curve[:, 0], curve[:, 1])] = 1

    kernel = np.ones((thickness, thickness), np.uint8)
    img = dilation(img, kernel)

    return img
