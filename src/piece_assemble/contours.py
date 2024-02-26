import cv2
import numpy as np

from piece_assemble.types import BinImg, Points


def extract_contours(img_bin: BinImg) -> tuple[Points, list[Points]]:
    """Extract contours from binary image.

    Parameters
    ----------
    img_bin
        A binary image of one piece which may or may not contain some holes.

    Returns
    -------
    outer_points
        Points belonging to the outer border of given piece.
        2d array of shape `[N, 2]` where rows are points `(y, x)`
    hole_points
        List of 2d arrays of points, each of them corresponds to one hole in the piece.
    """
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    hierarchy = hierarchy[0]

    outer_contour_i = np.where(hierarchy[:, 3] == -1)[0][0]
    outer_contour = contours[outer_contour_i]

    holes_contours = [
        contours[i] for i in np.where(hierarchy[:, 3] == outer_contour_i)[0]
    ]

    def contours_to_points(contour):
        return contour[:, 0, [1, 0]]

    return contours_to_points(outer_contour), [
        contours_to_points(contours) for contours in holes_contours
    ]
