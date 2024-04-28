from piece_assemble.geometry import Transformation, fit_transform
from piece_assemble.piece import Piece


def get_initial_transformation(
    piece1: Piece, piece2: Piece, idx1: int, idx2: int
) -> Transformation:
    points1 = piece1.segments[idx1].contour[[0, -1]]
    points2 = piece2.segments[idx2].contour[[-1, 0]]
    return fit_transform(points1, points2)
