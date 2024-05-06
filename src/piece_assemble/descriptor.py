from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from more_itertools import flatten

from piece_assemble.contours import get_osculating_circles, get_validity_intervals
from piece_assemble.geometry import (
    extend_interval,
    extend_intervals,
    interval_difference,
    points_dist,
)
from piece_assemble.segment import ApproximatingArc, Segment
from piece_assemble.types import NpImage, Point, Points

if TYPE_CHECKING:
    from piece_assemble.piece import Piece


class DescriptorExtractor(ABC):
    @abstractmethod
    def extract(
        self, contour: Points, image: NpImage
    ) -> tuple[list[Segment], np.ndarray]:
        return NotImplemented

    def dist(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return NotImplemented


class OsculatingCircleDescriptor(DescriptorExtractor):
    def __init__(
        self,
        n_points: int = 5,
        n_colors: int = 0,
        tol_dist: float = 2.5,
        channels: int = 3,
        min_segment_len: int = 5,
        spatial_dist_w: float = 0,
        color_dist_w: float = 10,
        color_var_w: float = 0,
        length_w: float = 0,
        rel_len_diff_w: float = 0,
        angle_w: float = 0,
    ):
        self.n_points = n_points
        self.n_colors = n_colors
        self.tol_dist = tol_dist
        self.channels = channels
        self.spatial_dist_w = spatial_dist_w
        self.color_dist_w = color_dist_w
        self.length_w = length_w
        self.rel_len_diff_w = rel_len_diff_w
        self.angle_w = angle_w
        self.min_segment_len = min_segment_len
        self.color_var_w = color_var_w
        self.spatial_dist_thr = 1000

    def extract(
        self, contour: Points, image: NpImage
    ) -> tuple[list[Segment], np.ndarray]:
        radii, centers = get_osculating_circles(contour)
        segments = approximate_curve_by_circles(contour, radii, centers, self.tol_dist)
        segments = [
            segment for segment in segments if len(segment) >= self.min_segment_len
        ]

        descriptor = np.array(
            [self.segment_descriptor(segment, image) for segment in segments]
        )

        return segments, descriptor

    def _get_points(self, segment: Segment, n_points: int) -> list[Point]:
        p_start = segment.contour[0]
        p_end = segment.contour[-1]

        subsegment_len = len(segment) // (n_points - 1)
        points = [
            p_start,
            *[segment.contour[subsegment_len * (i + 1)] for i in range(n_points - 2)],
            p_end,
        ]

        return points

    def segment_descriptor(
        self, segment: Segment, piece_img: NpImage
    ) -> np.ndarray[float]:
        """Get descriptor of given curve segment.

        Parameters
        ----------
        segment
            2d array of all points representing a contour segment.

        Returns
        -------
        A descriptor of contour segment - an array of 3 2d vectors.
        """
        centroid = segment.contour.mean(axis=0)
        p_start = segment.contour[0]
        p_end = segment.contour[-1]

        # Determine segment rotation
        rot_vector = p_start - p_end
        rot_vector = rot_vector / np.linalg.norm(rot_vector)

        sin_a = rot_vector[0]
        cos_a = rot_vector[1]
        rot_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

        vectors = self._get_points(segment, self.n_points)
        desc_vectors = [(vector - centroid) @ rot_matrix for vector in vectors]

        colors_points = self._get_points(segment, self.n_colors)
        desc_colors = [
            piece_img[round(point[0]), round(point[1])] for point in colors_points
        ]

        return np.concatenate(desc_vectors + desc_colors)

    def color_dist(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        dist = np.zeros((first.shape[0], second.shape[0]))

        for i in range(self.n_colors):
            dist += points_dist(
                first[
                    :,
                    self.n_points * 2
                    + i * self.channels : self.n_points * 2
                    + (i + 1) * self.channels,
                ],
                second[
                    :,
                    first.shape[1]
                    - (i + 1) * self.channels : first.shape[1]
                    - i * self.channels,
                ],
            )

        return dist / self.n_colors

    def color_var(self, desc: np.ndarray) -> float:
        vars = [
            np.var(
                desc[
                    :,
                    [
                        self.n_points * 2 + i * self.channels + ch
                        for i in range(self.n_colors)
                    ],
                ],
                axis=1,
            )
            for ch in range(self.channels)
        ]
        var = np.max(vars, axis=0)
        return var

    def spatial_dist(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        dist = np.zeros((first.shape[0], second.shape[0]))

        for i in range(self.n_points):
            dist += points_dist(
                first[:, i * 2 : (i + 1) * 2],
                -second[:, (self.n_points - i - 1) * 2 : (self.n_points - i) * 2],
            )

        dist = dist / self.n_points
        dist[dist > self.spatial_dist_thr] = np.inf
        return dist

    def dist(self, piece1: Piece, piece2: Piece) -> np.ndarray:
        color_dist = self.color_dist(piece1.descriptor, piece2.descriptor)
        spatial_dist = self.spatial_dist(piece1.descriptor, piece2.descriptor)
        min_var = np.minimum(
            self.color_var(piece1.descriptor)[:, np.newaxis],
            self.color_var(piece2.descriptor)[np.newaxis, :],
        )

        len1 = piece1.get_segment_lengths()
        len2 = piece2.get_segment_lengths()
        max_len = np.maximum(len1[:, np.newaxis], len2[np.newaxis, :])
        min_len = np.minimum(len1[:, np.newaxis], len2[np.newaxis, :])
        rel_len_diff = (max_len - min_len) / max_len

        radii1 = np.array([segment.radius for segment in piece1.segments])
        radii2 = np.array([segment.radius for segment in piece2.segments])
        angles1 = len1 / (2 * np.pi * radii1)
        angles2 = len2 / (2 * np.pi * radii2)

        min_angles = np.minimum(angles1[:, np.newaxis], angles2[np.newaxis, :])

        eps = 0.0000001
        return (
            spatial_dist * self.spatial_dist_w
            + color_dist * self.color_dist_w
            + self.color_var_w / (min_var + eps)
            + self.length_w / (max_len + eps)
            + rel_len_diff * self.rel_len_diff_w
            + self.angle_w / (min_angles + eps)
        )


class MultiOsculatingCircleDescriptor(OsculatingCircleDescriptor):
    def __init__(
        self,
        n_points: int = 5,
        n_colors: int = 0,
        tol_dists: list[float] = [2.5],
        channels: int = 3,
        min_segment_len: int = 5,
        spatial_dist_w: float = 0,
        color_dist_w: float = 10,
        color_var_w: float = 0,
        length_w: float = 0,
        rel_len_diff_w: float = 0,
        angle_w: float = 0,
    ):
        self.n_points = n_points
        self.n_colors = n_colors
        self.tol_dists = tol_dists
        self.channels = channels
        self.spatial_dist_w = spatial_dist_w
        self.color_dist_w = color_dist_w
        self.length_w = length_w
        self.rel_len_diff_w = rel_len_diff_w
        self.angle_w = angle_w
        self.min_segment_len = min_segment_len
        self.color_var_w = color_var_w
        self.spatial_dist_thr = 1000

    def extract(
        self, contour: Points, image: NpImage
    ) -> tuple[list[Segment], np.ndarray]:
        radii, centers = get_osculating_circles(contour)
        segments = list(
            flatten(
                [
                    approximate_curve_by_circles(contour, radii, centers, tol_dist)
                    for tol_dist in self.tol_dists
                ]
            )
        )
        segments = [
            segment for segment in segments if len(segment) >= self.min_segment_len
        ]

        descriptor = np.array(
            [self.segment_descriptor(segment, image) for segment in segments]
        )

        return segments, descriptor


def approximate_curve_by_circles(
    contour: Points, radii: np.ndarray[float], centers: Points, tol_dist: float
) -> list[ApproximatingArc]:
    """Obtain the curve approximation by osculating circle arcs.

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

    Returns
    -------
    circles
        I list of circle arc representations. Each element is a tuple
        `(i, validity_interval)`, where `i` is the index of osculating circle and
        `validity_interval` is a range of indexes where the given contour is well
        approximated by this circle.
    """
    validity_intervals = get_validity_intervals_split(contour, radii, centers, tol_dist)
    cycle_length = contour.shape[0]
    validity_intervals_extended = extend_intervals(validity_intervals, cycle_length)
    interval_indexes = np.arange(len(validity_intervals))

    # In each iteration, find the osculating circle with the largest validity interval.
    # Then, update all other intervals and repeat.
    arcs = []
    while True:
        valid_lengths = (
            validity_intervals_extended[:, 1] - validity_intervals_extended[:, 0]
        )
        max_i = np.argmax(valid_lengths)
        length = valid_lengths[max_i]
        if length <= 1:
            break
        validity_interval = validity_intervals[max_i]
        arcs.append((interval_indexes[max_i], validity_interval))
        validity_intervals = np.array(
            [
                interval_difference(r, validity_interval, cycle_length)
                for r in validity_intervals
            ]
        )
        # remove intervals of length 0:
        mask_is_nonzero = validity_intervals[:, 0] != validity_intervals[:, 1]
        validity_intervals = validity_intervals[mask_is_nonzero]
        interval_indexes = interval_indexes[mask_is_nonzero]

        validity_intervals_extended = extend_intervals(validity_intervals, cycle_length)

        if len(validity_intervals_extended) == 0:
            break

    arc_ordering = np.array([c[0] for c in arcs]).argsort()
    return [
        ApproximatingArc(
            arcs[i][1], contour, centers[arcs[i][0]], radii[arcs[i][0]], arcs[i][0]
        )
        for i in arc_ordering
    ]


def get_splitting_points(radii: np.ndarray, min_segment_length: int) -> np.ndarray:
    """Find points where the curve can be split.

    Those points are the global curvature maxima.

    Parameters
    ----------
    radii
        Array of radii.
    min_segment_length
        Minimal length of one part.

    Returns
    -------
    array of indexes where the curve can be split.

    """
    candidate_points = np.argsort(np.abs(radii))
    radii_sorted = np.abs(radii[candidate_points])
    candidate_points = candidate_points[radii_sorted < 15]

    selected_points = []
    while len(candidate_points) > 0:
        new_point = candidate_points[0]
        selected_points.append(new_point)
        candidate_points = candidate_points[
            (np.abs(candidate_points - new_point) >= min_segment_length)
            & (
                (
                    np.minimum(new_point, candidate_points)
                    + len(radii)
                    - np.maximum(new_point, candidate_points)
                )
                >= min_segment_length
            )
        ]

    return np.sort(np.array(selected_points))


def get_validity_intervals_split(
    contour: Points,
    radii: np.ndarray,
    centers: Points,
    tol_dist: float,
    min_segment_length: int = 500,
) -> np.ndarray:
    """Get approximation validity intervals for each osculating circle.

    This function first splits the curve to several separate parts and computes
    the validity intervals for each part separately.

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
    min_segment_length
        Minimal length of one part.

    Returns
    -------
    validity_intervals
        Array of validity intervals for each osculating circle.
    """
    splitting_points = get_splitting_points(radii, min_segment_length)
    if len(splitting_points) <= 1:
        return get_validity_intervals(contour, radii, centers, tol_dist, True)

    # Shift contour so it starts in the fist one of splitting points
    shift = -splitting_points[0]
    contour = np.roll(contour, shift, axis=0)
    radii = np.roll(radii, shift)
    centers = np.roll(centers, shift, axis=0)
    splitting_points += shift

    segment_ranges = [
        extend_interval(interval, len(radii))
        for interval in zip(splitting_points, np.roll(splitting_points, -1))
    ]
    validity_intervals = list(
        flatten(
            [
                get_validity_intervals(
                    contour[r[0] : r[1]],
                    radii[r[0] : r[1]],
                    centers[r[0] : r[1]],
                    tol_dist,
                    False,
                )
                + r[0]
                for r in segment_ranges
            ]
        )
    )
    validity_intervals = np.roll(validity_intervals, -shift, axis=0)
    return (validity_intervals - shift) % len(radii)
