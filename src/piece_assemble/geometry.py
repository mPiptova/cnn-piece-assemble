import numpy as np

from piece_assemble.types import Point, Points


def point_to_line_dist(points: Points, line_segment: tuple[Point, Point]):
    """Compute the distance of each of given points and a line.

    Parameters
    ----------
    points
        2d array of points.
    line_segment
        A line given by two points.

    Returns
    -------
    distances
        Array of distances from each points to the given line.
    """
    a, b = line_segment
    segment_length = np.linalg.norm(a - b)
    return (
        (b[0] - a[0]) * (a[1] - points[:, 1]) - (b[1] - a[1]) * (a[0] - points[:, 0])
    ) / segment_length


def points_dist(points1: Point, points2: Points) -> np.ndarray[float]:
    """Return distances of all point pairs from given set of points.

    Parameters
    ----------
    points1
        2d array where rows represent points (x, y).
    points2
        2d array where rows represent points (x, y).

    Returns
    -------
    distances
        2d array where distance of `points1[i]` and `points2[j]` is stored at position
        `[i, j]`.
    """
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)
