import numpy as np
from skimage.segmentation import flood_fill

from puzzle_generator.lines import (
    draw_curve,
    generate_random_line,
    interpolate_curve,
    perturbate_points,
    sample_points_on_line,
)


def divide_plane_by_curve(curve: np.ndarray, height: int, width: int) -> np.ndarray:
    """Divide a plane by a curve.

    Parameters
    ----------
    curve
        The curve to divide the plane by, represented as a numpy array of shape
        (num_points, 2) where each row contains the row and column coordinates
        of a point.
    width
        The width of the image.
    height
        The height of the image.

    Returns
    -------
    A numpy array of shape (height, width) containing the image with the plane divided
    by the curve.
    One component of the image has value 0 and the other part has value 1.

    """
    img = draw_curve(curve, height, width, 3)

    background = np.where(img == 0)
    img = flood_fill(img, (background[0][0], background[1][0]), 1)

    # In half the cases, choose the second partition
    if np.random.rand() > 0.5:
        img = 1 - img

    return img


def add_division_by_curve(img: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Use the image to add one level of division to the image.

    Parameters
    ----------
    img
        The image to add the division to.
    curve
        The curve to divide the plane by, represented as a numpy array of shape
        (num_points, 2) where each row contains the row and column coordinates
        of a point.

    Returns
    -------
    A numpy array of shape (height, width) containing the image with the plane divided
    by the curve.
    The input image is multiplied by the division image with random values.
    """

    new_division = divide_plane_by_curve(curve, img.shape[0], img.shape[1]).astype(
        float
    )
    value = np.random.rand() * 0.3 + 0.5
    new_division[new_division == 0] = value

    return img * new_division


def add_division_level(
    img: np.ndarray, num_samples: int, perturbation_strength: float
) -> np.ndarray:
    """Add one level of division to the image.

    Parameters
    ----------
    img
        The image to add the division to.
    num_samples
        The number of points to sample on one line.
    perturbation_strength
        The strength of random perturbation.

    Returns
    -------
    A numpy array of the same shape as the input image with the division added.
    Each component (representing one piece) has different float value.
    """
    height, width = img.shape[0], img.shape[1]
    p1, p2 = generate_random_line(height, width)

    points = sample_points_on_line(p1, p2, num_samples)
    perturbed_points = perturbate_points(points, perturbation_strength, height, width)
    curve = interpolate_curve(perturbed_points, max(height, width) * 2000)
    img = add_division_by_curve(img, curve)
    return img


def get_random_division(
    height: int,
    width: int,
    num_curves: int,
    num_samples: int,
    perturbation_strength: float,
) -> np.ndarray:
    """Generate random division of the image with specified shape.

    Parameters
    ----------
    height
        The height of the image.
    width
        The width of the image.
    num_lines
        The number of curves which will be used to divide the image.
    num_samples
        The number of points to sample on each line.
    perturbation_strength
        The strength of random perturbation.

    Returns
    -------
    Image divided into pieces. Each component (representing one piece) has different
    float value from 0 to 1.
    """
    img = np.ones((height, width), dtype=float)
    for _ in range(num_curves):
        img = add_division_level(img, num_samples, perturbation_strength)
        # Normalize the image values (for better visualization)
        values = np.unique(img)
        for i in range(len(values)):
            img[img == values[i]] = 1 / (len(values) + 1) * (i + 1)

    return img
