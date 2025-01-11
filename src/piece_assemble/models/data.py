from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from skimage.morphology import dilation

from piece_assemble.geometry import get_common_contour_idxs
from piece_assemble.types import Points
from piece_assemble.visualization import draw_contour

if TYPE_CHECKING:
    from piece_assemble.piece import TransformedPiece


def get_correspondence_matrix(
    t_piece1: TransformedPiece,
    t_piece2: TransformedPiece,
    tol: int = 5,
    dilation_size: int = 3,
) -> np.ndarray:
    idxs1_closest, idxs2 = get_common_contour_idxs(
        t_piece1.contour,
        t_piece2.contour,
        tol,
    )
    idxs2_closest, idxs1 = get_common_contour_idxs(
        t_piece2.contour,
        t_piece1.contour,
        tol,
    )
    mask1 = np.isin(idxs2, idxs2_closest)
    mask2 = np.isin(idxs1, idxs1_closest)

    idxs1_closest = idxs1_closest[mask1]
    idxs2 = idxs2[mask1]

    idxs2_closest = idxs2_closest[mask2]
    idxs1 = idxs1[mask2]

    similarity_matrix = np.zeros((len(t_piece1.contour), len(t_piece2.contour)))

    similarity_matrix[idxs1_closest, idxs2] = 1
    similarity_matrix[idxs1, idxs2_closest] = 1

    dilation_mask = np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
        ]
    )
    similarity_matrix = dilation(similarity_matrix, dilation_mask)

    if dilation_size > 0:
        similarity_matrix = dilation(
            similarity_matrix, np.ones((dilation_size, dilation_size))
        )

    return similarity_matrix


def contour_to_patches(contour: Points, window_size: int) -> list[np.ndarray]:
    padding_size = window_size // 2

    img_contour = np.ones((contour.max(axis=0) + 1), dtype="uint8")

    img_contour = draw_contour(contour, img_contour, 0)
    img_contour = np.pad(
        img_contour,
        ((padding_size, padding_size), (padding_size, padding_size)),
        mode="constant",
        constant_values=1,
    )
    img_contour = 1 - img_contour

    patches = [
        img_contour[
            point[0] : point[0] + 2 * padding_size + 1,
            point[1] : point[1] + 2 * padding_size + 1,
        ]
        for point in contour
    ]
    return patches


def img_to_patches(
    contour: Points, img: np.ndarray, window_size: int
) -> list[np.ndarray]:
    padding_size = window_size // 2

    padding = [(padding_size, padding_size), (padding_size, padding_size)]
    if len(img.shape) == 3:
        padding.append((0, 0))
    img = np.pad(img, padding, mode="constant", constant_values=1)

    patches = [
        img[
            point[0] : point[0] + 2 * padding_size + 1,
            point[1] : point[1] + 2 * padding_size + 1,
        ]
        for point in contour
    ]
    return patches
