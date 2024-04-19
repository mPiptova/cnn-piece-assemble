from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.spatial import KDTree
from shapely import transform

from piece_assemble.clustering import Cluster, ClusterScorer, TransformedPiece
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

    def to_cluster(
        self, scorer: ClusterScorer, border_dist_tol, self_intersection_tol
    ) -> Cluster:
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


class Matches:
    def __init__(self, matches: list[Match]) -> None:
        columns = ["key1", "key2", "dist", "match"]

        def match_to_row(match: Match):
            return (match.key1, match.key2, match.dist, match)

        rows = [match_to_row(match) for match in matches]
        matches_df = pd.DataFrame(rows, columns=columns)
        self._df = matches_df.sort_values("dist").reset_index()

    def _update_df(self):
        self._df = self._df[
            self._df["match"].apply(lambda m: m.valid or m.valid is None)
        ]
        self._df["dist"] = self._df["match"].apply(lambda m: m.dist)
        self._df = self._df.sort_values("dist", ignore_index=True).reset_index()

    def sample(self, n: int) -> list[Match]:
        samples = self._df.sample(len(self._df), weights=1 / (self._df["dist"] ** 2))
        matches = []
        for _, row in samples.iterrows():
            if len(matches) == n:
                break
            match = row["match"]
            match.verify(2)
            if match.valid:
                matches.append(match)

        self._update_df
        return matches

    def head(self, n: int) -> list[Match]:
        matches = []
        for _, row in self._df.iterrows():
            if len(matches) == n:
                break
            match = row["match"]
            match.verify(2)
            if match.valid:
                matches.append(match)

        self._update_df
        return matches
