from piece_assemble.geometry import Transformation, fit_transform
from piece_assemble.piece import Piece


def get_initial_transformation(
    piece1: Piece, piece2: Piece, idx1: int, idx2: int
) -> Transformation:
    interval1 = piece1.segments[idx1].interval
    interval2 = piece2.segments[idx2].interval

    points1 = piece1.contour[interval1, :]
    points2 = piece2.contour[interval2[::-1], :]

    return fit_transform(points1, points2)
