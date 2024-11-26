from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from piece_assemble.feature_extraction.base import FeatureExtractor, Features

if TYPE_CHECKING:
    from piece_assemble.matching.match import Match
    from piece_assemble.piece import Piece
    from piece_assemble.types import NpImage, Points


class DummyDescriptor(Features):
    def __init__(self) -> None:
        super().__init__()
        pass

    def get_complexity(self, idxs: np.ndarray) -> float:
        return 1


class DummyFeatureExtractor(FeatureExtractor):
    def extract(self, contour: Points, image: NpImage) -> Features:
        return DummyDescriptor()

    def dist(self, first: Piece, second: Piece) -> np.ndarray:
        return np.zeros((0, 0))

    def find_all_matches(
        self,
        pieces: list[Piece],
    ) -> list[Match]:
        return []
