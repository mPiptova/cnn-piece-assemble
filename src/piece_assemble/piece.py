from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely import Polygon, geometry
from skimage.filters import rank
from skimage.measure import approximate_polygon
from skimage.morphology import diamond, dilation, disk, erosion

from geometry import Transformation, extend_interval
from image import load_bin_img, load_img
from piece_assemble.contours import extract_contours, smooth_contours
from piece_assemble.types import Points

if TYPE_CHECKING:
    from piece_assemble.descriptor import (
        Descriptor,
        DescriptorExtractor,
        DummyDescriptorExtractor,
    )
    from piece_assemble.segment import ApproximatingArc
    from piece_assemble.types import BinImg, NpImage


class Piece:
    def __init__(
        self,
        name: str,
        img: NpImage,
        img_avg: NpImage,
        mask: BinImg,
        contour: Points,
        descriptor_extractor: DescriptorExtractor,
        descriptor: Descriptor,
        holes: list[Points],
        hole_descriptors: list[Descriptor],
        polygon: Polygon,
    ):
        self.name = name
        self.img = img
        self.img_avg = img_avg
        self.mask = mask
        self.contour = contour
        self.descriptor_extractor = descriptor_extractor
        self.descriptor = descriptor
        self.holes = holes
        self.polygon = polygon
        self.hole_descriptors = hole_descriptors

    @classmethod
    def from_image(
        cls,
        name: str,
        img: NpImage,
        mask: BinImg,
        descriptor_extractor: DescriptorExtractor,
        sigma: float = 5,
        polygon_approximation_tolerance: float = 3,
        img_mean_window_r: int = 3,
    ) -> Piece:
        """Create a piece representation from an image and mask.

        Parameters
        ----------
        name
            The ID of the piece.
        img
            The image of the piece.
        mask
            The binary mask of the piece.
        descriptor_extractor
            The descriptor extractor used to extract the descriptor of the piece.
        sigma
            The standard deviation of the Gaussian kernel used for smoothing the
            contours.
        polygon_approximation_tolerance
            The tolerance for the polygon approximation algorithm.
        img_mean_window_r
            The radius of the circular window used for computing the average image.

        Returns
        -------
        A Piece object.
        """

        # For averaging, use eroded mask for better behavior near contours
        mask_eroded = erosion(mask.astype(bool), diamond(1))
        footprint = disk(img_mean_window_r)
        img_int = (img * 255).astype("uint8")
        if len(img.shape) == 3:
            img_avg = (
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
            img_avg = rank.mean(img, footprint, mask=mask_eroded)

        # Dilate mask to compensate for natural erosion of pieces
        contours = extract_contours(dilation(mask, diamond(1)).astype("uint8"))
        outline_contour = contours[0]
        holes = contours[1]

        contour = smooth_contours(outline_contour, sigma)
        holes = [smooth_contours(hole, sigma) for hole in holes if len(hole) > 100]

        descriptor = descriptor_extractor.extract(contour, img_avg)

        polygon = cls._get_polygon_approximation(
            polygon_approximation_tolerance, contour, holes
        )
        hole_descriptors = cls._extract_hole_descriptors(holes, img_avg)

        return cls(
            name,
            img,
            img_avg,
            mask,
            contour,
            descriptor_extractor,
            descriptor,
            holes,
            hole_descriptors,
            polygon,
        )

    @classmethod
    def _get_polygon_approximation(
        cls,
        polygon_approximation_tolerance: float,
        contour: Points,
        holes: list[Points],
    ) -> None:
        polygon = geometry.Polygon(
            approximate_polygon(contour, polygon_approximation_tolerance)
        )
        hole_polygons = [
            geometry.Polygon(approximate_polygon(hole, polygon_approximation_tolerance))
            for hole in holes
        ]

        for hole_polygon in hole_polygons:
            polygon = polygon.difference(hole_polygon)

        return polygon

    @classmethod
    def _extract_hole_descriptors(cls, holes: list[Points], img_avg: NpImage) -> None:
        hole_descriptors = []

        for hole in holes:
            descriptor = cls.descriptor_extractor.extract(hole, img_avg)
            hole_descriptors.append(descriptor)

        return

    def get_segment_lengths(self) -> np.ndarray:
        def arc_len(arc: ApproximatingArc):
            extended_interval = extend_interval(arc.interval, len(self.contour))
            return extended_interval[1] - extended_interval[0]

        return np.array([arc_len(arc) for arc in self.descriptor.segments])

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

        new_arcs = [arc for arc in self.descriptor.segments if is_large_enough(arc)]
        if len(new_arcs) == len(self.descriptor.segments):
            return

        self.descriptor.segments = new_arcs
        self.descriptor = np.array(
            [self.segment_descriptor(arc.contour) for arc in self.descriptor.segments]
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
            self.descriptor.contour_segment_idxs[idxs], return_counts=True
        )

        arc_idxs = [
            idx
            for idx, count in zip(*unique_arc_idxs1)
            if idx != -1 and count > 0.7 * len(self.descriptor.segments[idx])
        ]

        return len(arc_idxs)


class TransformedPiece(Piece):
    def __init__(self, piece: Piece, transformation: Transformation) -> None:
        super().__init__(
            piece.name,
            piece.img,
            piece.img_avg,
            piece.mask,
            piece.contour,
            piece.descriptor_extractor,
            piece.descriptor,
            piece.holes,
            piece.hole_descriptors,
            piece.polygon,
        )
        self._piece = piece

        self.polygon = shapely.transform(piece.polygon, transformation.apply)
        self.contour = transformation.apply(piece.contour)
        self.transformation = transformation

    @property
    def original_contour(self) -> Points:
        return self._piece.contour

    def transform(self, transformation: Transformation) -> TransformedPiece:
        return TransformedPiece(
            self._piece, self.transformation.compose(transformation)
        )


def load_images(
    img_dir: str, scale: float = 1
) -> tuple[list[str], list[NpImage], list[BinImg]]:
    """Load all piece images from the given directory.

    Parameters
    ----------
    img_dir
        Path to the directory containing piece images.
        Two images are required for every image, one PNG binary mask with "_mask" suffix
        and one JPG image with piece itself, both need to have the same size.
    scale
        All returned images will be rescaled by this factor.

    Returns
    -------
    img_ids
        A list of image names.
    imgs
        A list of 3-channel images of pieces
    masks
        A list of binary piece masks.
    """
    img_ids = [
        name
        for name in os.listdir(img_dir)
        if not name.endswith("_mask.png")
        and name not in ["pieces.json", "original.png"]
    ]

    imgs = [load_img(os.path.join(img_dir, name), scale) for name in img_ids]
    img_ids = [name.split(".")[0] for name in img_ids]
    masks = [
        load_bin_img(os.path.join(img_dir, f"{name}_mask.png"), scale)
        for name in img_ids
    ]
    return img_ids, imgs, masks


def load_pieces(
    path: str, descriptor: DescriptorExtractor | None = None
) -> dict[Piece]:
    """
    Load pieces from the given directory.

    Parameters
    ----------
    path
        Path to the directory containing piece images.

    Returns
    -------
    pieces
        A dictionary of Piece objects.
    """
    if descriptor is None:
        descriptor = DummyDescriptorExtractor()

    img_ids, imgs, masks = load_images(path)

    return {
        img_ids[i]: Piece.from_image(img_ids[i], imgs[i], masks[i], descriptor, 0)
        for i in range(len(img_ids))
    }
