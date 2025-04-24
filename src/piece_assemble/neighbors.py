from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from piece_assemble.geometry import get_common_contour_idxs

if TYPE_CHECKING:
    from piece_assemble.piece import Piece, TransformedPiece


class NeighborClassifierBase(ABC):
    @abstractmethod
    def __call__(self, piece1: TransformedPiece, piece2: TransformedPiece) -> bool:
        ...


class BorderLengthNeighborClassifier(NeighborClassifierBase):
    def __init__(self, min_border_length: int, dist_tol: float) -> None:
        super().__init__()
        self.min_border_length = min_border_length
        self.dist_tol = dist_tol

    def __call__(self, piece1: TransformedPiece, piece2: TransformedPiece) -> bool:
        idxs, _ = longest_continuous_border(piece1, piece2, self.dist_tol)
        return len(idxs) > self.min_border_length


def longest_continuous_border(
    piece1: TransformedPiece, piece2: TransformedPiece, border_dist_tol: float
) -> tuple[np.ndarray, Piece | None]:

    """
    Find the longest continuous border between two pieces, given their transformations.

    Given two transformed pieces, find the longest continuous border between them,
    given by the longest sequence of points in both contours that are within
    a certain distance of each other.

    Parameters
    ----------
    piece1
        The first piece.
    piece2
        The second piece.
    border_dist_tol
        The maximum distance between points in the two contours for them to be
        considered as being on the same border.

    Returns
    -------
    The indices of the longest border and the corresponding piece.
    """
    idxs1, idxs2 = get_common_contour_idxs(
        piece1.contour,
        piece2.contour,
        border_dist_tol,
    )
    if idxs2 is None:
        return np.empty(0), None

    def get_longest_continuous_idxs(idxs: np.ndarray, piece: Piece) -> np.ndarray:
        idxs = np.concatenate((idxs, idxs + len(piece.contour)))
        idxs = longest_continuous_subsequence(np.unique(idxs))
        return idxs % len(piece.contour)

    idxs1 = get_longest_continuous_idxs(idxs1, piece1)
    idxs2 = get_longest_continuous_idxs(idxs2, piece2)

    if len(idxs1) > len(idxs2):
        return idxs1, piece1
    return idxs2, piece2


def longest_continuous_subsequence(sequence: np.array, max_diff: int = 2) -> np.array:
    mask = (sequence[1:] - sequence[:-1] < max_diff).astype(int)
    mask = np.pad(mask, (1, 1), "constant", constant_values=(0, 0))
    diff = mask[1:] - mask[:-1]
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return np.array([])
    idx_max = (ends - starts).argmax()
    return sequence[starts[idx_max] : ends[idx_max]]
