from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from shapely import transform

from piece_assemble.cluster import Cluster, ClusterScorer, TransformedPiece
from piece_assemble.geometry import Transformation, fit_transform, icp
from piece_assemble.piece import Piece


class Match:
    """Represents one match between two pieces."""

    def __init__(self, piece1: Piece, piece2: Piece, idx1: int, idx2: int, dist: float):
        self.dist = dist
        self.id1 = piece1.name
        self.id2 = piece2.name
        self.contour1 = piece1.contour
        self.contour2 = piece2.contour

        self.polygon1 = piece1.polygon
        self.polygon2 = piece2.polygon
        self.match_points = (
            piece1.segments[idx1].contour[[0, -1]],
            piece2.segments[idx2].contour[[-1, 0]],
        )

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
        return fit_transform(*self.match_points)

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
        smaller_area = min(self.polygon1.area, self.polygon2.area)
        polygon1 = transform(self.polygon1, lambda pol: transformation.apply(pol))
        return self.polygon2.intersection(polygon1).area / smaller_area

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
        if self._ios(self.initial_transformation) > ios_tol:
            return False
        return True

    def verify(
        self,
        dist_tol: float,
        ios_tol: float = 0.02,
        icp_max_iters: int = 30,
        icp_min_change: float = 0.5,
    ) -> Match | None:
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
        transformation = self.initial_transformation

        # If the intersection of the transformed polygons is too large, reject this
        # match and don't continue with the computation
        if self._ios(transformation) > max(0.1, ios_tol):
            return None

        transformation = icp(
            self.contour1,
            self.contour2,
            transformation,
            dist_tol,
            icp_max_iters,
            icp_min_change,
        )

        # Check the intersection area again, this time with more strict threshold
        if self._ios(transformation) > ios_tol:
            return None

        return CompactMatch(self.id1, self.id2, transformation)


@dataclass
class CompactMatch:
    """Compact match representation used to save time in parallel processing"""

    id1: str
    id2: str
    transformation: Transformation

    def to_cluster(
        self,
        scorer: ClusterScorer,
        border_dist_tol: float,
        self_intersection_tol: float,
        pieces_dict: dict[Piece],
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
        pieces_dict
            Dictionary of all pieces.

        Returns
        -------
        cluster
        """
        piece1 = pieces_dict[self.id1]
        piece2 = pieces_dict[self.id2]

        pieces = {
            self.id1: TransformedPiece(piece1, self.transformation),
            self.id2: TransformedPiece(piece2, Transformation.identity()),
        }
        return Cluster(
            pieces,
            scorer=scorer,
            border_dist_tol=border_dist_tol,
            self_intersection_tol=self_intersection_tol,
        )
