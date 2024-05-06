from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from scipy.spatial import KDTree
from shapely import transform

from piece_assemble.clustering import Cluster, ClusterScorer, TransformedPiece
from piece_assemble.geometry import Transformation, fit_transform, icp
from piece_assemble.matching.utils import get_initial_transformation
from piece_assemble.piece import Piece


@dataclass
class Match:
    """Represents one match between two pieces."""

    dist: float
    piece1: Piece
    piece2: Piece
    index1: int
    index2: int
    transformation: Transformation | None = None

    def __post_init__(self):
        self.key1 = self.piece1.name
        self.key2 = self.piece2.name
        self.valid = None

    @cached_property
    def initial_transformation(self) -> Transformation:
        """Returns initial (not finetuned) transformation between pieces.

        This returns only the transformation between the corresponding segments,
        without any finetuning. As a result, it may be imprecise, but sufficient
        for filtering out matches that are completely wrong (e.g. overlapping).

        Returns
        -------
        transformation
            Transformation which maps `piece1` to match `piece2`.
        """
        return get_initial_transformation(
            self.piece1, self.piece2, self.index1, self.index2
        )

    def _ios(self, transformation: Transformation) -> float:
        """Computes intersection over smaller of matched pieces.

        Parameters
        ----------
        transformation
            Transformation to be applied to the first piece.

        Returns
        -------
        ios
            Intersection area / area of the smaller of two pieces.
        """
        smaller_area = min(self.piece1.polygon.area, self.piece2.polygon.area)
        polygon1 = transform(self.piece1.polygon, lambda pol: transformation.apply(pol))
        return self.piece2.polygon.intersection(polygon1).area / smaller_area

    def is_initial_transform_valid(self, ios_tol: float = 0.1) -> bool:
        """Returns bool indicated whether the initial transform is valid.

        False indicates that the two pieces overlap too much to form a valid match.
        Only transformation mapping one segment to another is used, without any
        finetuning, therefore the result might be imprecise. However, it can
        be used to quickly filter out matches that are a way too off.

        Parameters
        ----------
        ios_tol
            A threshold value for intersection over smaller of transformed pieces.

        Returns
        -------
        is_valid
        """
        if self.valid is not None:
            return self.valid
        if self._ios(self.initial_transformation) > ios_tol:
            self.valid = False
            return False

    def verify(self, dist_tol: float, ios_tol: float = 0.02) -> Match | None:
        """Returns more precise Match or None if invalid.

        Parameters
        ----------
        dist_tol
            Distance tolerance. If the distance of two points is below this
            threshold, they are considered as matching perfectly.
        polygon_intersection_tol
            Intersection over union of transformed pieces tolerance.
            If IoS is above this threshold, the match is considered invalid.

        Returns
        -------
        match
            Match with more accurate transformation estimation or None, if invalid.
        """
        if self.valid is not None:
            return self
        transformation = self.initial_transformation

        # If the intersection of the transformed polygons is too large, reject this
        # match and don't continue with the computation
        if self._ios(transformation) > max(0.1, ios_tol):
            self.valid = False
            return

        transformation = icp(
            self.piece1.contour, self.piece2.contour, transformation, dist_tol * 4
        )

        transformed_contour = transformation.apply(self.piece1.contour)
        tree = KDTree(self.piece2.contour)

        nearest_dist, _ = tree.query(transformed_contour, k=1)
        near_mask = nearest_dist < dist_tol

        # Check the intersection area again, this time with more strict threshold
        if self._ios(transformation) > 0.02:
            self.valid = False
            return

        dist = nearest_dist[near_mask].mean()
        self.valid = True

        self.transformation = transformation
        self.dist = dist

        new_match = Match(
            dist, self.piece1, self.piece2, self.index1, self.index2, transformation
        )
        new_match.valid = True

        return new_match

    def to_cluster(
        self,
        scorer: ClusterScorer,
        border_dist_tol: float,
        self_intersection_tol: float,
    ) -> Cluster:
        """Converts Match to Cluster.

        Parameters
        ----------
        scorer
            ClusterScorer which computes the score of the cluster.
        border_dist_tol
            Distance tolerance. If the distance of two border points
            is below this threshold, the borders are considered touching
            at that point.
        self_intersection_tol
            Self-intersection tolerance. If the relative intersection of any two
            pieces is above this threshold, the cluster is considered invalid.
            (Some relaxation is done depending on the total number of pieces.)


        Returns
        -------
        cluster
        """
        pieces = {
            self.key1: TransformedPiece(self.piece1, self.transformation),
            self.key2: TransformedPiece(self.piece2, Transformation.identity()),
        }
        return Cluster(
            pieces,
            scorer=scorer,
            border_dist_tol=border_dist_tol,
            self_intersection_tol=self_intersection_tol,
        )


def get_initial_transformation(
    piece1: Piece, piece2: Piece, idx1: int, idx2: int
) -> Transformation:
    """Find initial transformation based only on segment match.

    Parameters
    ----------
    piece1
    piece2
    idx1
        Index of the matched segment of `piece1`
    idx2
        Index of the matched segment of `piece2`

    Returns
    -------
    transformation
        A transformation mapping `piece1.segments[idx1]` to `piece2.segments[idx2]`
    """
    points1 = piece1.segments[idx1].contour[[0, -1]]
    points2 = piece2.segments[idx2].contour[[-1, 0]]
    return fit_transform(points1, points2)
