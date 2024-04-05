from abc import ABC, abstractmethod

import numpy as np

from piece_assemble.contours import get_osculating_circles
from piece_assemble.geometry import points_dist
from piece_assemble.piece import approximate_curve_by_circles
from piece_assemble.segment import Segment
from piece_assemble.types import NpImage, Point, Points


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
        color_w: float = 10,
    ):
        self.n_points = n_points
        self.n_colors = n_colors
        self.tol_dist = tol_dist
        self.channels = channels
        self.color_w = color_w

    def extract(
        self, contour: Points, image: NpImage
    ) -> tuple[list[Segment], np.ndarray]:
        radii, centers = get_osculating_circles(contour)
        segments = approximate_curve_by_circles(contour, radii, centers, self.tol_dist)

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

    def spatial_dist(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        dist = np.zeros((first.shape[0], second.shape[0]))

        for i in range(self.n_points):
            dist += points_dist(
                first[:, i * 2 : (i + 1) * 2],
                -second[:, (self.n_points - i - 1) * 2 : (self.n_points - i) * 2],
            )

        return dist / self.n_points

    def dist(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return self.spatial_dist(first, second) + self.color_w * self.color_dist(
            first, second
        )
