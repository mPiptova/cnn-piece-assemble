from __future__ import annotations

from typing import Generator

import numpy as np

from geometry import Transformation
from piece_assemble.cluster import Cluster, ClusterScorerBase
from piece_assemble.matching.match import Match
from piece_assemble.piece import Piece, TransformedPiece

Graph = np.ndarray[bool]


class TransformationGraph:
    def __init__(
        self,
        all_ids: list[str],
        transformation_dict: dict[tuple[str, str], Transformation],
    ) -> None:
        self.transformation_dict = transformation_dict
        self.all_ids = all_ids

        self.idx_map = {piece_id: i for i, piece_id in enumerate(all_ids)}

        self._adj_matrix = np.zeros((len(all_ids), len(all_ids)), dtype=bool)
        for i, j in transformation_dict.keys():
            self._adj_matrix[self.idx_map[i], self.idx_map[j]] = True
            self._adj_matrix[self.idx_map[j], self.idx_map[i]] = True

    @classmethod
    def from_matches(cls, matches: list[Match]) -> TransformationGraph:
        transformation_dict = {}
        for match in matches:
            if match is None:
                continue
            transformation_dict[(match.id1, match.id2)] = match.transformation
            transformation_dict[(match.id2, match.id1)] = match.transformation.inverse()

        all_ids = list(set().union(*transformation_dict.keys()))
        return cls(all_ids, transformation_dict)

    def _get_successors_i(self, i: int) -> np.array:
        return np.flatnonzero(self._adj_matrix[i])

    def _find_cycles_recursive(
        self, length: int, cycle: list[int]
    ) -> Generator[list[int], None, None]:
        successors = self._get_successors_i(cycle[-1])
        if len(cycle) == length:
            if cycle[0] in successors:
                yield cycle
        elif len(cycle) < length:
            for v in successors:
                if v in cycle:
                    continue
                yield from self._find_cycles_recursive(length, cycle + [v])

    def _make_cycles_unique(self, cycles: list[list[str]]) -> list[list[str]]:
        def get_all_equivalent_cycles(
            cycle: list[str],
        ) -> Generator[list[str], None, None]:
            for c in (cycle, cycle[::-1]):
                for i in range(len(c)):
                    yield c[i:] + c[:i]

        unique_cycles = []
        for cycle in cycles:
            is_unique = True
            for c in get_all_equivalent_cycles(cycle):
                if c in unique_cycles:
                    is_unique = False
                    break
            if is_unique:
                unique_cycles.append(cycle)

        return unique_cycles

    def find_cycles(self, length: int, unique: bool = True) -> list[list[str]]:
        all_cycles = []
        for v in range(self._adj_matrix.shape[0]):
            all_cycles.extend(self._find_cycles_recursive(length, [v]))

        def cycle_idxs_to_ids(cycle_idxs: list[int]) -> list[str]:
            return [self.all_ids[i] for i in cycle_idxs]

        all_cycles = [
            cycle_idxs_to_ids(cycle) for cycle in all_cycles  # type: ignore[arg-type]
        ]
        if not unique:
            return all_cycles

        return self._make_cycles_unique(all_cycles)

    def _is_cycle_consistent(
        self, cycle: list[str], angle_tol: float = 0.17, translation_tol: float = 30
    ) -> bool:
        t = Transformation.identity()
        for i in range(len(cycle)):
            t = self.transformation_dict[cycle[(i + 1) % len(cycle)], cycle[i]].compose(
                t
            )

        is_consistent: bool = t.is_close(
            Transformation.identity(), angle_tol, translation_tol
        )
        return is_consistent

    def find_consistent_cycles(
        self, length: int, angle_tol: float = 0.17, translation_tol: float = 30
    ) -> list[list[str]]:
        cycles = self.find_cycles(length, False)

        consistent_cycles = [
            cycle
            for cycle in cycles
            if self._is_cycle_consistent(cycle, angle_tol, translation_tol)
        ]

        return self._make_cycles_unique(consistent_cycles)

    def cycle_to_cluster(
        self,
        cycle: list[str],
        pieces: dict[str, Piece],
        scorer: ClusterScorerBase,
        config: dict,
    ) -> Cluster | None:
        t_pieces = {}
        t = Transformation.identity()
        for i in range(len(cycle)):
            t_pieces[cycle[i]] = TransformedPiece(pieces[cycle[i]], t)
            t = self.transformation_dict[cycle[(i + 1) % len(cycle)], cycle[i]].compose(
                t
            )

        cluster = Cluster(t_pieces, scorer=scorer, **config)
        if cluster.self_intersection > 0.01:
            return None
        if len(cluster.get_neighbor_pairs()) != len(cycle):
            return None
        return cluster
