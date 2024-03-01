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


def extend_interval(interval: Interval, cycle_length: int) -> Interval:
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
    if interval[1] < interval[0]:
        return (interval[0], interval[1] + cycle_length)
    return interval


def interval_difference(interval1, interval2, cycle_length):
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

    interval1 = extend_interval(interval1, cycle_length)
    interval2 = extend_interval(interval2, cycle_length)

    cropped_range = (0, 0)
    if interval2[1] <= interval1[0] or interval1[1] <= interval2[0]:
        # they do not overlap
        cropped_range = interval1
    elif interval2[0] <= interval1[0] and interval2[1] >= interval1[1]:
        # interval1 is contained in interval2
        cropped_range = (interval1[0], interval1[0])
    elif interval1[0] < interval2[0]:
        cropped_range = interval1[0], interval2[0]
    else:
        cropped_range = (interval2[1], interval1[1])

    return normalize_interval(cropped_range, cycle_length)
