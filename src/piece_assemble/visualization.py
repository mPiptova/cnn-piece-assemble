import numpy as np
from PIL import ImageDraw
from PIL.Image import Image as PilImage

from piece_assemble.image import np_to_pil
from piece_assemble.piece import ApproximatingArc
from piece_assemble.types import NpImage, Point, Points


def draw_contour(
    contour: Points, contours_img: NpImage = None, value: float = 0
) -> NpImage:
    """Draw contours in an image represented as numpy array

    Parameters
    ----------
    contour
        2d array of all points representing a shape contour.
    contours_img
        Image represented as numpy array where the contours will be drawn.
        If None, new array with the ideal shape will be created.
    value
        Color of drawn contours. 0 corresponds to black, 1 to white.

    Returns
    -------
    Image represented as numpy array.
    """
    contour = np.round(contour).astype(int)

    # If image is not given, create empty one
    if contours_img is None:
        # Shift coordinates to eliminate empty space
        min_coordinates = np.min(contour, axis=0)
        contour = contour - min_coordinates
        contours_img = np.ones((contour[:, 0].max() + 1, contour[:, 1].max() + 1))
    contours_img[contour[:, 0], contour[:, 1]] = value
    return contours_img


def draw_circle_arc(
    radius: float,
    center: Point,
    draw: ImageDraw,
    color: int = 128,
    angle_range: tuple[int, int] = None,
) -> None:
    """Draw a circle arc given by radius, center and an angle range.

    Parameters
    ----------
    radius
        Radius of the circle.
    center
        Center of the circle, given as an array [row_coordinate, col_coordinate]
    draw
        ImageDraw instance where the arc should be drawn.
    color
        Outline color, 0 is black, 255 is white.
    angle_range
        Specifies which part of the circle should be drawn. A tuple of two angles in
        degrees.
        If None, whole circle is drawn.
    """
    leftUpPoint = (center[1] - radius, center[0] - radius)
    rightDownPoint = (center[1] + radius, center[0] + radius)
    twoPointList = [leftUpPoint, rightDownPoint]
    if angle_range is None:
        draw.ellipse(twoPointList, outline=color)
    else:
        draw.arc(twoPointList, start=angle_range[0], end=angle_range[1], fill=color)


def draw_circle_approximation(
    contour: Points,
    circle_arcs: list[ApproximatingArc],
) -> PilImage:
    """Draw a curve approximated by circle arcs.

    Parameters
    ----------
    contour
        2d array of all points representing a shape contour.
    circle_arcs
        Description of approximating circle arcs.
        A list of tuples `(i, (start, end))` where `i` is the index of a point in
        `contour`, to which the approximating osculating circle belongs to.
        `(start, end)` describes which part of `contour` is well approximated by this
        arc.
    centers
        Centers of all osculating circles of `contour` points.
    radii
        Radii of all osculating circles of `contour` points.

    Returns
    -------
    An image of a given curve approximated by osculating circles.

    """
    img = np.ones((int(contour[:, 0].max()) + 1, int(contour[:, 1].max()) + 1))
    img = np_to_pil(img)
    draw = ImageDraw.Draw(img)

    for arc in circle_arcs:
        # center = centers[i]
        # radius = np.abs(radii[i])
        point_start, point_end = (
            contour[arc.interval[0]],
            contour[arc.interval[1]],
        )

        alpha_start = np.rad2deg(np.arctan2(*(point_start - arc.center))) % 360
        alpha_end = np.rad2deg(np.arctan2(*(point_end - arc.center))) % 360

        if arc.radius > 0:
            angle_range = (alpha_end, alpha_start)
        else:
            angle_range = (alpha_start, alpha_end)

        draw_circle_arc(
            np.abs(arc.radius), arc.center, draw, color=100, angle_range=angle_range
        )
    return img
