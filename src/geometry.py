from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import KDTree

from piece_assemble.types import Interval, Point, Points


def point_to_line_dist(points: Points, line_segment: tuple[Point, Point]) -> np.ndarray:
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


def extend_intervals(intervals: np.ndarray, cycle_length: int) -> np.ndarray:
    """Extend the intervals in the cyclic domain.

    This function converts intervals such as (9, 2) to (9, 12), given that
    `cycle_length == 10`. In the output interval, the fist bound is always smaller than
    the second one.

    Parameters
    ----------
    interval
        2d array, each row represents a range.
    cycle_length
        Total length of the cycle domain where the intervals lie.

    Returns
    -------
    extended_intervals
        An interval where on of the numbers might be < 0 or >= `cycle_lengths`, but the
        first bound is always smaller than the second one.
    """
    return np.hstack(
        (
            intervals[:, 0, np.newaxis],
            np.where(
                intervals[:, 0] <= intervals[:, 1],
                intervals[:, 1],
                intervals[:, 1] + cycle_length,
            )[:, np.newaxis],
        )
    )


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
        return interval[0] <= num <= interval[1]  # type: ignore
    return num >= interval[0] or num <= interval[1]  # type: ignore


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

    if np.all(interval1 == interval2):
        return interval1[0], interval1[0]

    if is_in_cyclic_interval(
        interval2[0], interval1, cycle_length
    ) and is_in_cyclic_interval(interval2[1], interval1, cycle_length):
        return (interval1[0], interval1[0])

    if is_in_cyclic_interval(interval2[0], interval1, cycle_length):
        return (interval1[0], interval2[0])

    if is_in_cyclic_interval(interval2[1], interval1, cycle_length):
        return (interval2[1], interval1[1])

    if is_in_cyclic_interval(
        interval1[0], interval2, cycle_length
    ) and is_in_cyclic_interval(interval1[1], interval2, cycle_length):
        return (interval1[0], interval1[0])

    return interval1


def fit_transform(
    points1: Points, points2: Points, use_ransac: bool = False
) -> Transformation:
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
    transformation
    """
    b = points2.flatten()
    a = np.vstack([[[p[0], -p[1], 1, 0], [p[1], p[0], 0, 1]] for p in points1])

    if use_ransac:
        from skimage.measure import ransac
        from skimage.transform import EuclideanTransform

        model, _ = ransac((points1, points2), EuclideanTransform, 2, 5)

        if not np.isnan(model.rotation):
            return Transformation(-model.rotation, model.translation)

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

    angle = np.arctan2(-rot_matrix[0, 1], rot_matrix[0, 0])
    return Transformation(angle, translation)


def draw_line_polar(
    image: np.ndarray, line: tuple[float, float], thickness: int = 1
) -> None:
    """
    Draw a line defined in polar coordinates on the given image.

    Parameters
    ----------
    image
        Image to draw on
    line
        Tuple (rho, theta) of polar coordinates
    thickness
        Thickness of the line
    """
    rho, theta = line

    max_distance = 2 * max(image.shape)

    # Convert polar coordinates to Cartesian coordinates
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    # Calculate the endpoints of the line
    x1 = int(x0 + max_distance * (-b))
    y1 = int(y0 + max_distance * (a))
    x2 = int(x0 - max_distance * (-b))
    y2 = int(y0 - max_distance * (a))

    # Draw the line on the image
    cv2.line(image, (x1, y1), (x2, y2), 1, thickness)


def icp(
    points1: Points,
    points2: Points,
    init_transformation: Transformation,
    dist_tol: float = 20,
    max_iters: int = 30,
    min_change: float = 0.5,
) -> Transformation:
    """Find transformation between two sets of points using Iterative Closest Point
    algorithm

    Finds rotation and translation which maps `points1` to `points2`. This function
    expects the two cloud of points to already be reasonably close to each other.

    Parameters
    ----------
    points1
        The first point cloud. 2d array of points.
    points2
        The second point cloud. 2d array of points.
    init_transformation
        Initial transform to be applied to `points1`.

    Returns
    -------
    transformation
        Transformation which maps `points1` to `points2`
    """
    tree = KDTree(points2)
    points_transformed = init_transformation.apply(points1)

    total_transformation = init_transformation
    for _ in range(max_iters):
        t = icp_iteration(points_transformed, tree, dist_tol)
        total_transformation = total_transformation.compose(t)
        contours_new = t.apply(points_transformed)
        if np.linalg.norm(contours_new - points_transformed, axis=1).max() < min_change:
            break
        points_transformed = contours_new

    return total_transformation


def icp_iteration(
    points1: Points, points2_tree: KDTree, dist_tol: float = 20
) -> Transformation:
    """Do one iteration of the Iterative Closest Point algorithm.

    Parameters
    ----------
    points1
        The first point cloud. 2d array of points.
    points2_tree
        KDTree of points from the second point cloud.

    Returns
    -------
    transformation
    """
    nearest_dist, nearest_ind = points2_tree.query(points1, k=1)
    near_idx = nearest_dist < dist_tol

    if near_idx.sum() == 0:
        # objects are too far from each other, no transformation can be found
        return Transformation(0, np.array([0, 0]))
    points1_near = points1[near_idx]
    points2_near = points2_tree.data[nearest_ind][near_idx]

    return fit_transform(points1_near, points2_near)


def get_rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


@dataclass
class Transformation:
    """Class for representing geometric transformation,
    in particular rotation and translation.
    """

    rotation_angle: float
    translation: Point

    def __post_init__(self) -> None:
        self.rotation_angle = self.rotation_angle % (2 * np.pi)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return transformation rotation matrix."""
        return get_rotation_matrix(self.rotation_angle)

    def apply(self, points: Points) -> Points:
        """Apply transformation to a set of points

        Parameters
        ----------
        points
            2d array of points.

        Returns
        -------
        2d array of rotated points.
        """
        return points @ self.rotation_matrix + self.translation

    def compose(self, second: Transformation) -> Transformation:
        """Compose this transformation with another one.

        This transformation is applied first, the other second.

        Parameters
        ----------
        second
            A transformation that should be applied second.

        Returns
        -------
        Composed transformation.
        """
        angle = self.rotation_angle + second.rotation_angle
        translation = self.translation @ second.rotation_matrix + second.translation

        return Transformation(angle, translation)

    @classmethod
    def identity(cls) -> Transformation:
        """Return identity transformation."""
        return cls(0, np.array([0, 0]))

    def inverse(self) -> Transformation:
        new_translation = -self.translation @ get_rotation_matrix(-self.rotation_angle)
        return Transformation(-self.rotation_angle, new_translation)

    def is_close(
        self,
        other: Transformation,
        angle_tol: float = 0.17,
        translation_tol: float = 21,
    ) -> bool:
        return (
            abs(self.rotation_angle - other.rotation_angle) <= angle_tol
            and np.linalg.norm(self.translation - other.translation) <= translation_tol
        )

    def to_dict(self) -> dict:
        """Return the transformation parameters as a dictionary."""
        return {
            "rotation_angle": self.rotation_angle,
            "translation": list(self.translation),
        }

    @classmethod
    def from_dict(cls, params: dict) -> Transformation:
        """Create transfromation from parameters stored in dictionary."""
        return cls(params["rotation_angle"], np.array(params["translation"]))


def get_common_contour_idxs(
    contour1: Points, contour2: Points, tol: float = 10
) -> tuple[np.ndarray, np.ndarray]:
    tree1 = KDTree(contour1)
    distances, points = tree1.query(contour2, k=1)
    # Return indexes of points which are close enough
    close_mask = distances < tol
    return points[close_mask], np.where(close_mask)[0]


def get_common_contour(
    contour1: Points, contour2: Points, tol: float = 10
) -> tuple[Points, Points]:
    tree1 = KDTree(contour1)
    distances, points = tree1.query(contour2, k=1)
    # Return points which are close enough
    close_mask = distances < tol
    return contour1[points[close_mask]], contour2[close_mask]


def get_common_contour_length(
    contour1: Points, contour2: Points, tol: float = 10
) -> int:
    return len(get_common_contour(contour1, contour2, tol)[0])
