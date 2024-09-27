import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk
from skimage.segmentation import flood_fill

from puzzle_generator.lines import (
    draw_curve,
    generate_random_line,
    interpolate_curve,
    perturbate_points,
    sample_points_on_line,
)


def divide_plane_by_curve(curve: np.ndarray, height: int, width: int) -> np.ndarray:
    """Divide a plane by a curve.

    Parameters
    ----------
    curve
        The curve to divide the plane by, represented as a numpy array of shape
        (num_points, 2) where each row contains the row and column coordinates
        of a point.
    width
        The width of the image.
    height
        The height of the image.

    Returns
    -------
    A numpy array of shape (height, width) containing the image with the plane divided
    by the curve.
    One component of the image has value 0 and the other part has value 1.

    """
    img = draw_curve(curve, height, width, 3)

    background = np.where(img == 0)
    img = flood_fill(img, (background[0][0], background[1][0]), 1)

    # In half the cases, choose the second partition
    if np.random.rand() > 0.5:
        img = 1 - img

    return img


def add_division_by_curve(img: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Use the image to add one level of division to the image.

    Parameters
    ----------
    img
        The image to add the division to.
    curve
        The curve to divide the plane by, represented as a numpy array of shape
        (num_points, 2) where each row contains the row and column coordinates
        of a point.

    Returns
    -------
    A numpy array of shape (height, width) containing the image with the plane divided
    by the curve.
    The input image is multiplied by the division image with random values.
    """

    new_division = divide_plane_by_curve(curve, img.shape[0], img.shape[1]).astype(
        float
    )
    value = np.random.rand() * 0.3 + 0.5
    new_division[new_division == 0] = value

    return img * new_division


def add_division_level(
    img: np.ndarray, num_samples: int, perturbation_strength: float
) -> np.ndarray:
    """Add one level of division to the image.

    Parameters
    ----------
    img
        The image to add the division to.
    num_samples
        The number of points to sample on one line.
    perturbation_strength
        The strength of random perturbation.

    Returns
    -------
    A numpy array of the same shape as the input image with the division added.
    Each component (representing one piece) has different float value.
    """
    height, width = img.shape[0], img.shape[1]
    p1, p2 = generate_random_line(height, width)

    points = sample_points_on_line(p1, p2, num_samples)
    perturbed_points = perturbate_points(points, perturbation_strength, height, width)
    curve = interpolate_curve(perturbed_points, max(height, width) * 2000)
    img = add_division_by_curve(img, curve)
    return img


def get_random_division(
    height: int,
    width: int,
    num_curves: int,
    num_samples: int,
    perturbation_strength: float,
) -> np.ndarray:
    """Generate random division of the image with specified shape.

    Parameters
    ----------
    height
        The height of the image.
    width
        The width of the image.
    num_lines
        The number of curves which will be used to divide the image.
    num_samples
        The number of points to sample on each line.
    perturbation_strength
        The strength of random perturbation.

    Returns
    -------
    Image divided into pieces. Each component (representing one piece) has different
    float value from 0 to 1.
    """
    img = np.ones((height, width), dtype=float)
    for _ in range(num_curves):
        img = add_division_level(img, num_samples, perturbation_strength)
        img = _to_float_division_labels(img)

    return img


def _to_int_division_labels(division: np.ndarray) -> np.ndarray:
    int_division = np.zeros_like(division, dtype=int)
    values = np.unique(division)
    for i in range(len(values)):
        int_division[division == values[i]] = i + 1

    return int_division


def _to_float_division_labels(division: np.ndarray) -> np.ndarray:
    float_division = np.zeros_like(division, dtype=float)
    values = np.unique(division)
    for i in range(len(values)):
        float_division[division == values[i]] = 1 / (len(values) + 1) * (i + 1)

    return float_division


def reduce_number_of_pieces(
    division: np.ndarray, num_pieces: int, min_piece_area: int
) -> np.ndarray:
    """Reduce the number of pieces in the division.

    Parameters
    ----------
    division
        The division to reduce. Image, where each piece is represented by distinct
        float value from 0 to 1.
    num_pieces
        The desired maximal number of pieces.
    min_piece_area
        The minimum area of a piece.

    Returns
    -------
    A numpy array containing the reduced division.
    """
    # Change the type of division to int
    division = _to_int_division_labels(division)

    labels = label(division, connectivity=1)
    label_props = regionprops(labels)
    label_props.sort(key=lambda lprop: lprop["area"])

    previous_count = len(label_props) + 1
    while previous_count != len(label_props) and (
        len(label_props) > num_pieces or label_props[0]["area"] < min_piece_area
    ):
        previous_count = len(label_props)

        # choose the smallest piece
        piece = label_props[0]
        bbox = piece["bbox"]

        mask = np.zeros_like(labels, dtype=bool)
        mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = piece["image"]
        dil_mask = dilation(mask, disk(1)).astype(bool)
        dil_mask[mask] = False

        # find label of neighbor with the largest border witch chosen piece
        neighbor_labels = np.unique(labels[dil_mask], return_counts=True)
        neighbor_label = neighbor_labels[0][np.argmax(neighbor_labels[1])]

        # change the label
        labels[mask] = neighbor_label

        # update label properties
        label_props = regionprops(labels)
        label_props.sort(key=lambda lprop: lprop["area"])

    return labels


def get_puzzle_division(
    height: int,
    width: int,
    num_pieces: int,
    min_piece_area: int,
    num_curves: int,
    num_samples: int,
    perturbation_strength: float,
) -> np.ndarray:
    """Generate puzzle division of the image with specified shape.

    Parameters
    ----------
    height
        The height of the image.
    width
        The width of the image.
    num_pieces
        The number of pieces.
    min_piece_area
        The minimum area of a piece.
    num_curves
        The number of curves which will be used to divide the image.
    num_samples
        The number of points to sample on each line.
    perturbation_strength
        The strength of random perturbation.

    Returns
    -------
    Image divided into pieces. Each component (representing one piece) has different
    integer value.
    """
    division = get_random_division(
        height, width, num_curves, num_samples, perturbation_strength
    )
    division = reduce_number_of_pieces(division, num_pieces, min_piece_area)
    division = _to_int_division_labels(division)
    return division
