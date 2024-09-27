import numpy as np
from skimage.segmentation import flood_fill

from puzzle_generator.lines import draw_curve


def divide_plane_by_curve(curve: np.ndarray, width: int, height: int) -> np.ndarray:
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
    img = draw_curve(curve, width, height, 3)

    background = np.where(img == 0)
    img = flood_fill(img, (background[0][0], background[1][0]), 1)

    return img
