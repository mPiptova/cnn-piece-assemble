from __future__ import annotations

import numpy as np
from shapely import geometry
from skimage.filters import rank
from skimage.measure import approximate_polygon
from skimage.morphology import diamond, disk, erosion

from piece_assemble.contours import extract_contours, smooth_contours
from piece_assemble.descriptor import DescriptorExtractor
from piece_assemble.geometry import extend_interval
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
        if len(self.img.shape) == 3:
            self.img_avg = (
                np.stack(
                    [
                        rank.mean(self.img[:, :, channel], footprint, mask=mask_eroded)
                        for channel in range(self.img.shape[2])
                    ],
                    axis=2,
                )
                / 255
            )
        else:
            self.img_avg = rank.mean(self.img, footprint, mask=mask_eroded)

        contour = extract_contours(mask)[0]
        self.contour = smooth_contours(contour, sigma)
        self.polygon = geometry.Polygon(
            approximate_polygon(self.contour, polygon_approximation_tolerance)
        )

        self.segments, self.descriptor = self.descriptor_extractor.extract(
            self.contour, self.img_avg
        )

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
