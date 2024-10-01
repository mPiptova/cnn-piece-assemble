from __future__ import annotations

import numpy as np
from shapely import geometry
from skimage.filters import rank
from skimage.measure import approximate_polygon
from skimage.morphology import diamond, dilation, disk, erosion

from geometry import Transformation, extend_interval
from piece_assemble.contours import extract_contours, smooth_contours
from piece_assemble.descriptor import DescriptorExtractor
from piece_assemble.segment import ApproximatingArc
from piece_assemble.types import BinImg, NpImage


class Piece:
    def __init__(
        self,
        name: str,
        img: NpImage,
        mask: BinImg,
        descriptor_extractor: DescriptorExtractor,
        sigma: float = 5,
        polygon_approximation_tolerance: float = 3,
        img_mean_window_r: int = 3,
    ) -> None:

        self.name = name
        self.img = img
        self.mask = mask
        self.descriptor_extractor = descriptor_extractor

        # For averaging, use eroded mask for better behavior near contours
        mask_eroded = erosion(self.mask.astype(bool), diamond(1))
        footprint = disk(img_mean_window_r)
        img_int = (self.img * 255).astype("uint8")
        if len(self.img.shape) == 3:
            self.img_avg = (
                np.stack(
                    [
                        rank.mean(img_int[:, :, channel], footprint, mask=mask_eroded)
                        for channel in range(3)
                    ],
                    axis=2,
                )
                / 255
            )
        else:
            self.img_avg = rank.mean(self.img, footprint, mask=mask_eroded)

        # Dilate mask to compensate for natural erosion of pieces
        contours = extract_contours(dilation(self.mask, diamond(1)).astype("uint8"))
        outline_contour = contours[0]
        holes = contours[1]

        self.contour = smooth_contours(outline_contour, sigma)
        self.holes = [smooth_contours(hole, sigma) for hole in holes if len(hole) > 100]

        self.segments, self.descriptor = self.descriptor_extractor.extract(
            self.contour, self.img_avg
        )

        self._get_polygon_approximation(polygon_approximation_tolerance)

        self.contour_segment_idxs = np.full(len(self.contour), -1)
        for i, segment in enumerate(self.segments):
            if segment.interval[0] < segment.interval[1]:
                self.contour_segment_idxs[segment.interval[0] : segment.interval[1]] = i
            else:
                self.contour_segment_idxs[segment.interval[0] :] = i
                self.contour_segment_idxs[: segment.interval[1]] = i

        self._extract_hole_descriptors()

    def _get_polygon_approximation(
        self, polygon_approximation_tolerance: float
    ) -> None:
        self.polygon = geometry.Polygon(
            approximate_polygon(self.contour, polygon_approximation_tolerance)
        )
        hole_polygons = [
            geometry.Polygon(approximate_polygon(hole, polygon_approximation_tolerance))
            for hole in self.holes
        ]

        for hole_polygon in hole_polygons:
            self.polygon = self.polygon.difference(hole_polygon)

    def _extract_hole_descriptors(self) -> None:
        hole_segments = []
        hole_descriptors = []

        for hole in self.holes:
            segments, descriptor = self.descriptor_extractor.extract(hole, self.img_avg)
            hole_segments.append(segments)
            hole_descriptors.append(descriptor)
            for segment in segments:
                segment.offset = len(self.contour)

            self.descriptor = np.concatenate([self.descriptor, descriptor])
            self.contour = np.concatenate([self.contour, hole])

            hole_segment_idxs = np.full(len(hole), -1)
            for i, segment in enumerate(segments):
                if segment.interval[0] < segment.interval[1]:
                    hole_segment_idxs[segment.interval[0] : segment.interval[1]] = i
                else:
                    hole_segment_idxs[segment.interval[0] :] = i
                    hole_segment_idxs[: segment.interval[1]] = i
            hole_segment_idxs += len(self.segments)
            self.contour_segment_idxs = np.concatenate(
                [self.contour_segment_idxs, hole_segment_idxs]
            )

            self.segments.extend(segments)

    def get_segment_lengths(self) -> np.ndarray:
        def arc_len(arc: ApproximatingArc):
            extended_interval = extend_interval(arc.interval, len(self.contour))
            return extended_interval[1] - extended_interval[0]

        return np.array([arc_len(arc) for arc in self.segments])

    def filter_small_arcs(self, min_size: float, min_angle: float) -> None:
        """Filter out circle arcs which are too small.

        Parameters
        ----------
        min_size
            Circle arcs with size larger than this number won't be filtered out.
        min_angle
            Circle arcs with the size smaller than `min_size`, but angle larger than
            `min_angle` won't be filtered out.
            The angle is given in radians.
        """

        def is_large_enough(arc: ApproximatingArc) -> bool:
            if (
                np.linalg.norm(
                    self.contour[arc.interval[0]] - self.contour[arc.interval[1]]
                )
                >= min_size
            ):
                return True
            length = len(arc)
            return length >= np.abs(arc.radius) * min_angle

        new_arcs = [arc for arc in self.segments if is_large_enough(arc)]
        if len(new_arcs) == len(self.segments):
            return

        self.segments = new_arcs
        self.descriptor = np.array(
            [self.segment_descriptor(arc.contour) for arc in self.segments]
        )

    def get_segment_count(self, idxs: np.ndarray) -> int:
        """Return the number of segments included in the given contour selection.

        Parameters
        ----------
        idxs
            Indexes defining the contour section.


        Returns
        -------
        segment_count
            The total number of segments that the border section spans over.
        """
        unique_arc_idxs1 = np.unique(
            self.contour_segment_idxs[idxs], return_counts=True
        )

        arc_idxs = [
            idx
            for idx, count in zip(*unique_arc_idxs1)
            if idx != -1 and count > 0.7 * len(self.segments[idx])
        ]

        return len(arc_idxs)


class TransformedPiece:
    def __init__(self, piece: Piece, transformation: Transformation) -> None:
        self.piece = piece
        self.transformation = transformation

    def transform(self, transformation: Transformation) -> TransformedPiece:
        return TransformedPiece(self.piece, self.transformation.compose(transformation))
