import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from piece_assemble.types import BinImg, Points


def extract_contours(img_bin: BinImg) -> tuple[Points, list[Points]]:
    """Extract contours from binary image.

    Parameters
    ----------
    img_bin
        A binary image of one piece which may or may not contain some holes.

    Returns
    -------
    outer_points
        Points belonging to the outer border of given piece.
        2d array of shape `[N, 2]` where rows are points `(y, x)`
    hole_points
        List of 2d arrays of points, each of them corresponds to one hole in the piece.
    """
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    hierarchy = hierarchy[0]

    outer_contour_i = np.where(hierarchy[:, 3] == -1)[0][0]
    outer_contour = contours[outer_contour_i]

    holes_contours = [
        contours[i] for i in np.where(hierarchy[:, 3] == outer_contour_i)[0]
    ]

    def contours_to_points(contour):
        return contour[:, 0, [1, 0]]

    return contours_to_points(outer_contour), [
        contours_to_points(contours) for contours in holes_contours
    ]


def smooth_contours(contours: Points, sigma: float) -> Points:
    """Smooth contour curve with gaussian filter.

    Parameters
    ----------
    contours
        2d array of points
    sigma
        Size of the gaussian filter

    Returns
    -------
    smoothed_contours
        2d array of points
    """
    return np.stack(
        [
            gaussian_filter1d(contours[:, 0].astype(float), sigma, mode="wrap"),
            gaussian_filter1d(contours[:, 1].astype(float), sigma, mode="wrap"),
        ],
        axis=1,
    )


def diff(f: np.ndarray) -> np.ndarray:
    """Approximate first derivative of function `f`.


    Parameters
    ----------
    f
        Cyclic array of function values.

    Returns
    -------
    df
    """
    return np.roll(f, -1) - f


def changes_sign(f: np.ndarray) -> np.ndarray[int]:
    """Find indexes where the sign of given function changes.

    Parameters
    ----------
    f
        Cyclic array of function values.

    Returns
    -------
    indexes
        Indexes where the sign changes.
    """
    sign = np.where(f == 0, 0, f // np.abs(f))
    return np.where((diff(sign) != 0) | (sign == 0))[0]


def find_inflection_points(contour: Points) -> np.ndarray[int]:
    """Find indexes of inflection points in given contour points.

    Parameters
    ----------
    contour
        2d array of points representing a shape contour.

    Returns
    -------
    indexes
        Array of indexes of inflection points.
    """
    dx1 = diff(contour[:, 0])
    dy1 = diff(contour[:, 1])

    dx2 = diff(dx1)
    dy2 = diff(dy1)

    return changes_sign(dx1 * dy2 + dy1 * dx2)


def find_curvature_extrema(contour: Points) -> np.ndarray[int]:
    """Find indexes of contour where the curvature extrema are reached.

    Parameters
    ----------
    contour
        2d array of points representing a shape contour.

    Returns
    -------
    indexes
        Array of indexes of points where the curvature extrema are reached.
    """
    dx1 = diff(contour[:, 0])
    dy1 = diff(contour[:, 1])

    dx2 = diff(dx1)
    dy2 = diff(dy1)

    K_numerator = dx1 * dy2 + dy1 * dx2
    K_denominator = np.power(dx1 * dx1 + dy1 * dy1, 3 / 2)

    K = K_numerator / K_denominator

    K_minima_idxs = argrelextrema(K, np.less)[0]
    K_maxima_idxs = argrelextrema(K, np.greater)[0]
    return np.sort(np.concatenate((K_minima_idxs, K_maxima_idxs)))
