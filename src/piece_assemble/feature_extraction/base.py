from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from piece_assemble.matching.match import Match
    from piece_assemble.piece import Piece
    from piece_assemble.types import BinImg, NpImage, Points


class Features(ABC):
    @abstractmethod
    def get_complexity(self, idxs: np.ndarray) -> float:
        return NotImplemented


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, contour: Points, image: NpImage) -> Features:
        return NotImplemented

    @abstractmethod
    def dist(self, first: Piece, second: Piece) -> np.ndarray:
        return NotImplemented

    @abstractmethod
    def find_all_matches(
        self,
        pieces: list[Piece],
    ) -> list[Match]:
        return NotImplemented

    @abstractmethod
    def prepare_image(
        self, image: NpImage, mask: BinImg, blur_image: NpImage
    ) -> NpImage:
        return NotImplemented
