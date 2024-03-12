import numpy as np

from piece_assemble.types import Interval, Point, Points


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


def normalize_interval(interval: Interval, cycle_length: int) -> Interval:
    """Normalize interval within the cycle domain.

    When we work with cycles (e.g. closed curves), it's sometimes beneficial to work
    with intervals such as (-1, 2) or (9 , 12) instead of (9, 2), where the cycle length
    is 10. This methods converts the interval to the standard representation, where both
    bounds are in the range `[0, cycle_length]`.

    Parameters
    ----------
    interval
        Pair of numbers representing a range.
    cycle_length
        Total length of the cycle domain where the interval lies.

    Returns
    -------
    normalized_interval
        Interval where all numbers are in range `[0, cycle_length]`, but the first
        bound may be greater than the second one.
    """
    return interval[0] % cycle_length, interval[1] % cycle_length


def extend_interval(
    interval: Interval, cycle_length: int, keep_bound: int = 0
) -> Interval:
    """Extend the interval in the cyclic domain.

    This function converts intervals such as (9, 2) to (9, 12), given that
    `cycle_length == 10`. In the output interval, the fist bound is always smaller than
    the second one.

    Parameters
    ----------
    interval
        Pair of numbers representing a range.
    cycle_length
        Total length of the cycle domain where the interval lies.

    Returns
    -------
    extended_interval
        An interval where on of the numbers might be < 0 or >= `cycle_lengths`, but the
        first bound is always smaller than the second one.
    """
    if interval[0] <= interval[1]:
        return interval
    if keep_bound == 0:
        return (interval[0], interval[1] + cycle_length)
    return (interval[0] - cycle_length, interval[1])


def is_in_cyclic_interval(num: float, interval: Interval, cycle_length: int) -> bool:
    """Determine whether the given number is in the cyclic interval.

    Parameters
    ----------
    num
    interval
        Pair of numbers representing a range.
    cycle_length
        Total length of the cycle domain where the interval lie.

    Returns
    -------
    bool
    """
    if num < 0 or num >= cycle_length:
        num = num % cycle_length

    interval = normalize_interval(interval, cycle_length)

    if interval[0] < interval[1]:
        return interval[0] <= num <= interval[1]
    return num >= interval[0] or num <= interval[1]


def interval_difference(
    interval1: Interval, interval2: Interval, cycle_length: int
) -> Interval:
    """Crop the first interval so it doesn't intersect the second one.

    Parameters
    ----------
    interval1
        Pair of numbers representing a range.
    interval2
        Pair of numbers representing a range. This interval will be subtracted from the
        first one.
    cycle_length
        Total length of the cycle domain where the intervals lie.

    Returns
    -------
    cropped_interval
        A set difference of given intervals.
    """
    if interval1[0] == interval1[1]:
        return interval1

    if interval1 == interval2:
        return interval1[0], interval1[0]

    if is_in_cyclic_interval(
        interval2[0], interval1, cycle_length
    ) and is_in_cyclic_interval(interval2[1], interval1, cycle_length):
        # complicated case, not supported
        print(interval1, interval2)
        raise ValueError("Not supported")

    if is_in_cyclic_interval(interval2[0], interval1, cycle_length):
        return (interval1[0], interval2[0])

    if is_in_cyclic_interval(interval2[1], interval1, cycle_length):
        return (interval2[1], interval1[1])

    if is_in_cyclic_interval(
        interval1[0], interval2, cycle_length
    ) and is_in_cyclic_interval(interval1[1], interval2, cycle_length):
        return (interval1[0], interval1[0])

    return interval1


def fit_transform(points1: Points, points2: Points) -> tuple[np.ndarray, Point]:
    """Find transformation which transforms one set of point into another.

    This transformation includes only rotation and translation, not scaling or shearing.

    Parameters
    ----------
    points1
        2d array of points.
    points2
        2d array of points.

    Returns
    -------
    rot_matrix
        A transformation (2, 2) matrix.
    translation
        2d translation vector.
    """
    b = points2.flatten()
    a = np.vstack([[[p[0], -p[1], 1, 0], [p[1], p[0], 0, 1]] for p in points1])

    x = np.linalg.lstsq(a, b, rcond=None)[0]

    rot_matrix = np.array(
        [
            [x[0], x[1]],
            [-x[1], x[0]],
        ]
    )
    scale = np.linalg.norm(rot_matrix[0])
    rot_matrix = rot_matrix / scale

    points1_t = points1 @ rot_matrix

    translation = np.mean(points2, axis=0) - np.mean(points1_t, axis=0)

    return rot_matrix, translation
