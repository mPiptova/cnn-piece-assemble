from __future__ import annotations

from functools import cached_property
from itertools import combinations

import shapely

from piece_assemble.geometry import Transformation
from piece_assemble.osculating_circle_descriptor import OsculatingCircleDescriptor


class Cluster:
    def __init__(
        self, pieces: dict[str, tuple[OsculatingCircleDescriptor, Transformation]]
    ) -> None:
        self._pieces = pieces

        self.descriptors = {key: desc for key, (desc, _) in pieces.items()}
        self.transformations = {key: t for key, (_, t) in pieces.items()}

    @cached_property
    def border_length(self) -> int:
        # TODO: Implement this
        return 0

    @property
    def piece_ids(self) -> set[str]:
        return set(self._pieces.keys())

    def copy(self):
        return Cluster(self._pieces.copy())

    def transform(self, transformation: Transformation) -> Cluster:
        new_pieces = {
            key: (desc, t.compose(transformation))
            for key, (desc, t) in self._pieces.items()
        }
        return Cluster(new_pieces)

    @cached_property
    def self_intersection(self) -> float:
        polygons = [
            shapely.transform(desc._polygon, lambda pol: t.apply(pol))
            for desc, t in self._pieces.values()
        ]
        return max(
            [
                p1.intersection(p2).area / min(p1.area, p2.area)
                for p1, p2 in combinations(polygons, 2)
            ]
        )

    def intersection(self, polygon) -> float:
        polygons = [
            shapely.transform(desc._polygon, lambda pol: t.apply(pol))
            for desc, t in self._pieces.values()
        ]
        return max(
            [p.intersection(polygon).area / min(p.area, polygon.area) for p in polygons]
        )

    def merge(self, other: Cluster) -> Cluster:
        common_keys = self.piece_ids.intersection(other.piece_ids)

        common_key = common_keys.pop()
        cluster1 = self.transform(self.transformations[common_key].inverse())
        cluster2 = other.transform(other.transformations[common_key].inverse())

        for key in common_keys:
            if not cluster1.transformations[key].is_close(
                cluster2.transformations[key]
            ):
                return None

        new_cluster = Cluster(cluster1._pieces.update(cluster2._pieces))
        if new_cluster.self_intersection > 0.1:
            return None
        return new_cluster

    @cached_property
    def score(self) -> float:
        # TODO: More sensible score
        return len(self._pieces.keys())
