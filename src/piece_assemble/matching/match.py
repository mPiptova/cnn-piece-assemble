from __future__ import annotations

from dataclasses import dataclass

from scipy.spatial import KDTree
from shapely import transform

from piece_assemble.clustering import Cluster
from piece_assemble.geometry import Transformation, icp
from piece_assemble.matching.utils import get_initial_transformation
from piece_assemble.piece import Piece


@dataclass
class Match:
    key1: str
    key2: str
    dist: float
    piece1: Piece
    piece2: Piece
    index1: int
    index2: int
    transformation: Transformation | None = None
    score: float | None = None
    valid: bool | None = None

    def is_similar(
        self, other: Match, angle_tol: float = 0.17, translation_tol: float = 15
    ) -> bool:
        if self.key1 != other.key1 or self.key2 != other.key2:
            return False

        return self.transformation.is_close(
            other.transformation, angle_tol, translation_tol
        )

    def verify(self, dist_tol: float = 4):
        if self.valid is not None:
            return
        transformation = get_initial_transformation(
            self.piece1, self.piece2, self.index1, self.index2
        )

        # If the intersection of the transformed polygons is too large, reject this
        # match and don't continue with the computation
        smaller_area = min(self.piece1.polygon.area, self.piece2.polygon.area)
        polygon1 = transform(self.piece1.polygon, lambda pol: transformation.apply(pol))
        if self.piece2.polygon.intersection(polygon1).area / smaller_area > 0.1:
            self.valid = False
            return

        transformation = icp(
            self.piece1.contour, self.piece2.contour, transformation, dist_tol * 4
        )

        transformed_contour = transformation.apply(self.piece1.contour)
        tree = KDTree(self.piece2.contour)

        nearest_dist, _ = tree.query(transformed_contour, k=1)
        near_mask = nearest_dist < dist_tol

        length = near_mask.sum()

        # Check the intersection area again, this time with more strict threshold
        polygon1 = transform(self.piece1.polygon, lambda pol: transformation.apply(pol))
        if self.piece2.polygon.intersection(polygon1).area > length * 4 * dist_tol:
            self.valid = False
            return

        self.valid = True
        self.score = length
        self.transformation = transformation
        self.dist = nearest_dist[near_mask].mean() / (10 * length)

    def to_cluster(self) -> Cluster:
        pieces = {
            self.key1: (self.piece1, self.transformation),
            self.key2: (self.piece2, Transformation.identity()),
        }
        return Cluster(pieces)
