from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from piece_assemble.geometry import normalize_interval, point_to_line_dist, points_dist
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

    def contours_to_points(contour: Sequence) -> Points:
        return contour[:, 0, [1, 0]]

    return contours_to_points(outer_contour), [
        contours_to_points(contours) for contours in holes_contours
    ]


def smooth_contours(contours: Points, sigma: float, cyclic: bool = True) -> Points:
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
    if sigma == 0:
        return contours

    mode = "wrap" if cyclic else "reflect"
    return np.stack(
        [
            gaussian_filter1d(contours[:, 0].astype(float), sigma, mode=mode),
            gaussian_filter1d(contours[:, 1].astype(float), sigma, mode=mode),
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
    sign = np.where(f == 0, 1, f // np.abs(f))
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


def compute_curvature(contour: Points) -> np.ndarray[float]:
    """Return the curvature of given function.

    Parameters
    ----------
    contour
        2d array of points representing a closed shape contour.

    Returns
    -------
    curvature
        2d array of contour curvature in given points.
    """
    dx1 = diff(contour[:, 0])
    dy1 = diff(contour[:, 1])

    dx2 = diff(dx1)
    dy2 = diff(dy1)

    K_numerator = dx1 * dy2 - dy1 * dx2
    K_denominator = np.power(dx1 * dx1 + dy1 * dy1, 3 / 2)

    return K_numerator / K_denominator


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
    K = compute_curvature(contour)

    K_minima_idxs = argrelextrema(K, np.less)[0]
    K_maxima_idxs = argrelextrema(K, np.greater)[0]
    return np.sort(np.concatenate((K_minima_idxs, K_maxima_idxs)))


def split_interest_points(
    interest_point_idxs: np.ndarray[int], contour: Points, thr: float
) -> np.ndarray[int]:
    """Add new interest points if current interest points are not dense enough.

    Interest points divide the contour into a set of open curve segments. For two
    consecutive interest points `contour[i]` and `contour[j]`, this segment is given
    by points `contour[i+1:j]`. From these points, let `contour[k]` be the most distant
    point from the line given by points `contour[i]` and `contour[j]`.
    If this distance is above the selected threshold, `contour[k]` is added as a new
    interest point.

    Parameters
    ----------
    interest_point_idxs
        An array of indexes of interest points within the `all_points` array.
    contour
        2d array of all points representing a shape contour.
    thr
        A distance threshold used to determine if the segment between two consecutive
        interest points needs to be divided in two.

    Returns
    -------
    New array of interest points indexes.
    """
    new_idxs = []
    for start, end in zip(interest_point_idxs, np.roll(interest_point_idxs, -1)):
        inner_idx = (
            np.arange(start + 1, end)
            if start < end
            else np.arange(start + 1, len(contour) + end) % len(contour)
        )

        if len(inner_idx) == 0:
            continue

        dists = np.abs(
            point_to_line_dist(contour[inner_idx], (contour[start], contour[end]))
        )
        max_idx = dists.argmax()
        if dists[max_idx] > thr:
            new_idxs.append(inner_idx[max_idx])

    return np.sort(np.concatenate([interest_point_idxs, np.array(new_idxs)])).astype(
        int
    )


def merge_interest_points(
    interest_point_idxs: np.ndarray,
    all_points: Points,
    thr: float,
    allow_self_crossing: bool = False,
) -> np.ndarray:
    """Remove interest points if current interest points are too dense.


    Parameters
    ----------
    interest_point_idxs
        An array of indexes of interest points within the `all_points` array.
    contour
        2d array of all points representing a shape contour.
    thr
        A distance threshold used to determine if the segment between two consecutive
        interest points needs to be merged.

    Returns
    -------
    New array of interest points indexes.
    """
    idxs_to_remove = []
    start = interest_point_idxs[0]
    for middle, end in zip(
        np.roll(interest_point_idxs, -1), np.roll(interest_point_idxs, -2)
    ):
        inner_idx = (
            np.arange(start + 1, end)
            if start < end
            else np.arange(start + 1, len(all_points) + end) % len(all_points)
        )

        dists = point_to_line_dist(
            all_points[inner_idx], (all_points[start], all_points[end])
        )
        abs_dists = np.abs(dists)
        max_idx = abs_dists.argmax()
        if abs_dists[max_idx] < thr and (
            allow_self_crossing or np.abs(dists.sum()) == abs_dists.sum()
        ):
            idxs_to_remove.append(middle)
        else:
            start = middle

    return np.setdiff1d(interest_point_idxs, np.array(idxs_to_remove))


def get_osculating_circles(contours: Points) -> tuple[np.ndarray[float], Points]:
    """Find radius and center of osculating circle at any given point of curve.

    Parameters
    ----------
    contours
        2d array of points representing a closed shape contour.

    Returns
    -------
    radii
        1d array of circle radii.
    centers
        2d array of center points.
    """
    curvature = compute_curvature(contours)
    dx = diff(contours[:, 0])
    dy = diff(contours[:, 1])

    normals = np.vstack([-dy, dx]).T
    normals = normals / np.expand_dims(np.linalg.norm(normals, axis=1), 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        radii = 1 / curvature
        centers = contours + np.expand_dims(radii, 1) * normals

    return radii, centers


def get_validity_intervals(
    contour: Points,
    radii: np.ndarray[float],
    centers: Points,
    tol_dist: float,
    closed_curve: bool = True,
) -> np.ndarray:
    """Return the validity intervals for each osculating circle.

    Parameters
    ----------
    contour
        2d array of all points representing a shape contour.
    radii
        1d array of osculating circle radii.
    centers
        2d array of osculating circle center points.
    tol_dist
        Distance tolerance. If the distance of the contour point and the osculating
        circle is smaller than this number, the point is in the validity interval of
        this circle.
    closed_curve
        Whether the curve is closed.

    Returns
    -------
    validity_interval
        A list of (normalized) validity intervals for each osculating circle.
    """
    if len(contour) == 0:
        return []
    dists = points_dist(contour, centers)
    with np.errstate(invalid="ignore"):
        dists_from_circle = np.abs(dists - np.abs(radii))
    valid = (dists_from_circle < tol_dist).astype(int)
    valid_changes = np.roll(valid, -1, axis=0) - valid

    if not closed_curve:
        valid_changes[0, :] = 1
        valid_changes[-1, :] = -1

    starts_x, starts_y = np.where(valid_changes == 1)
    if closed_curve:
        starts_y = np.tile(starts_y, 2)
        starts_x = np.concatenate((-(contour.shape[0] - starts_x), starts_x))

    ends_x, ends_y = np.where(valid_changes == -1)
    if closed_curve:
        ends_y = np.tile(ends_y, 2)
        ends_x = np.concatenate((ends_x, contour.shape[0] + ends_x))

    validity_intervals = [
        (
            np.max(starts_x[(starts_y == i) & (starts_x <= i)]),
            np.min(ends_x[(ends_y == i) & (ends_x >= i)]) + 1,
        )
        if not np.isinf(radii[i])
        and len(starts_x[(starts_y == i) & (starts_x <= i)]) != 0
        and len(ends_x[(ends_y == i) & (ends_x >= i)]) != 0
        else (i, i + 1)
        for i in range(len(radii))
    ]

    if closed_curve:
        validity_intervals = [
            normalize_interval(interval, len(radii)) for interval in validity_intervals
        ]

    return np.array(validity_intervals)
