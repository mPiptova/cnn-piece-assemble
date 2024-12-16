from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely import Polygon, geometry, make_valid
from skimage.filters import rank
from skimage.measure import approximate_polygon
from skimage.morphology import diamond, dilation, disk, erosion

from geometry import Transformation
from piece_assemble.contours import extract_contours, smooth_contours
from piece_assemble.types import Points

if TYPE_CHECKING:
    from piece_assemble.feature_extraction.base import FeatureExtractor, Features
    from piece_assemble.types import BinImg, NpImage


class Piece:
    def __init__(
        self,
        name: str,
        img: NpImage,
        img_avg: NpImage,
        mask: BinImg,
        contour: Points,
        feature_extractor: FeatureExtractor,
        features: Features,
        holes: list[Points],
        hole_features: list[Features] | None,
        polygon: Polygon,
    ):
        self.name = name
        self.img = img
        self.img_avg = img_avg
        self.mask = mask
        self.contour = contour
        self.feature_extractor = feature_extractor
        self.features = features
        self.holes = holes
        self.polygon = polygon
        self.hole_features = hole_features

    @classmethod
    def from_image(
        cls,
        name: str,
        img: NpImage,
        mask: BinImg,
        feature_extractor: FeatureExtractor,
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
        feature_extractor
            The feature extractor used to extract the features of the piece.
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
        mask = mask.astype(bool)
        # For averaging, use eroded mask for better behavior near contours
        mask_eroded = erosion(mask, diamond(1))
        footprint = disk(img_mean_window_r)
        img_int = (img * 255).astype("uint8")
        img_avg = img
        if img_mean_window_r != 0:
            if len(img.shape) == 3:
                img_avg = (
                    np.stack(
                        [
                            rank.mean(
                                img_int[:, :, channel], footprint, mask=mask_eroded
                            )
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

        features = feature_extractor.extract(
            contour, feature_extractor.prepare_image(img, mask, img_avg)
        )

        polygon = cls._get_polygon_approximation(
            polygon_approximation_tolerance, contour, holes
        )
        polygon = make_valid(polygon)
        hole_features = cls._extract_hole_features(holes, img_avg)

        return cls(
            name,
            img,
            img_avg,
            mask,
            contour,
            feature_extractor,
            features,
            holes,
            hole_features,
            polygon,
        )

    @classmethod
    def _get_polygon_approximation(
        cls,
        polygon_approximation_tolerance: float,
        contour: Points,
        holes: list[Points],
    ) -> Polygon:
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
    def _extract_hole_features(
        cls, holes: list[Points], img_avg: NpImage
    ) -> list[Features]:
        hole_features = []

        for hole in holes:
            feature = cls.feature_extractor.extract(hole, img_avg)
            hole_features.append(feature)

        return hole_features

    def to_piece(self) -> Piece:
        return self


class TransformedPiece(Piece):
    def __init__(self, piece: Piece, transformation: Transformation) -> None:
        super().__init__(
            piece.name,
            piece.img,
            piece.img_avg,
            piece.mask,
            piece.contour,
            piece.feature_extractor,
            piece.features,
            piece.holes,
            piece.hole_features,
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

    def to_piece(self) -> Piece:
        return self._piece
