from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from piece_assemble.types import NpImage, Points


def draw_contour(
    contour: Points, contours_img: NpImage = None, value: float = 0
) -> NpImage:
    """Draw contours in an image represented as numpy array

    Parameters
    ----------
    contour
        2d array of all points representing a shape contour.
    contours_img
        Image represented as numpy array where the contours will be drawn.
        If None, new array with the ideal shape will be created.
    value
        Color of drawn contours. 0 corresponds to black, 1 to white.

    Returns
    -------
    Image represented as numpy array.
    """
    contour = np.round(contour).astype(int)

    # If image is not given, create empty one
    if contours_img is None:
        # Shift coordinates to eliminate empty space
        min_coordinates = np.min(contour, axis=0)
        contour = contour - min_coordinates
        contours_img = np.ones((contour[:, 0].max() + 1, contour[:, 1].max() + 1))
    contours_img[contour[:, 0], contour[:, 1]] = value
    return contours_img
