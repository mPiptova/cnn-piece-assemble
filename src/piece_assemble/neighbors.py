from abc import ABC, abstractmethod

import numpy as np

from geometry import get_common_contour_idxs
from piece_assemble.contours import smooth_contours
from piece_assemble.piece import Piece, TransformedPiece
from piece_assemble.types import Points
from piece_assemble.utils import longest_continuous_subsequence


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


class ComplexityNeighborClassifier(NeighborClassifierBase):
    def __init__(self, dist_tol: float, min_complexity: float = 1) -> None:
        super().__init__()
        self.dist_tol = dist_tol
        self.min_complexity = min_complexity

    def __call__(self, piece1: TransformedPiece, piece2: TransformedPiece) -> bool:
        return (
            get_border_complexity(piece1, piece2, self.dist_tol) > self.min_complexity
        )


def get_curve_winding_angle(curve: Points) -> float:
    """
    Calculate the total angle of turn of a curve.

    The total angle of turn of a curve is the sum of all the angles between
    consecutive points in the curve, divided by 2 pi. The result is a float
    between 0 and 1, where 0 means the curve is a straight line and 1 means
    the curve makes a full turn.

    Parameters
    ----------
    curve
        The points of the curve.

    Returns
    -------
    The total angle of turn of the curve.
    """
    curve = smooth_contours(curve, 3, False)
    curve_diff = curve[:-1] - curve[1:]

    if len(curve_diff) == 0:
        return 0

    curve_angle = np.arctan2(curve_diff[:, 0], curve_diff[:, 1])
    curve_angle = np.unwrap(curve_angle, discont=np.pi)
    return (curve_angle.max() - curve_angle.min()) / (2 * np.pi)


def get_border_complexity(
    piece1: TransformedPiece, piece2: TransformedPiece, border_dist_tol: float
) -> bool:
    """
    Calculate the complexity of a border between two pieces.

    The complexity of a border is calculated as the product of the segment count
    of the border and the winding angle of the border. The winding angle is
    the total angle of turn of the curve, divided by 2 pi.

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
    complexity
        The complexity of the border.
    """
    idxs, piece = longest_continuous_border(piece1, piece2, border_dist_tol)

    if len(idxs) == 0:
        return 0

    segment_count = piece.get_segment_count(idxs)
    winding_angle = get_curve_winding_angle(piece.contour[idxs])

    return segment_count * winding_angle


def longest_continuous_border(
    piece1: TransformedPiece, piece2: TransformedPiece, border_dist_tol: float
) -> tuple[np.ndarray, Piece]:

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
        piece1.transformation.apply(piece1.piece.contour),
        piece2.transformation.apply(piece2.piece.contour),
        border_dist_tol,
    )
    if idxs2 is None:
        return [], None

    def get_longest_continuous_idxs(idxs: np.ndarray, piece: Piece):
        idxs = np.concatenate((idxs, idxs + len(piece.contour)))
        idxs = longest_continuous_subsequence(np.unique(idxs))
        return idxs % len(piece.contour)

    idxs1 = get_longest_continuous_idxs(idxs1, piece1.piece)
    idxs2 = get_longest_continuous_idxs(idxs2, piece2.piece)

    if len(idxs1) > len(idxs2):
        return idxs1, piece1.piece
    return idxs2, piece2.piece
