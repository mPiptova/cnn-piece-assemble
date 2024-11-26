from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from geometry import Transformation
from image import load_bin_img, load_img
from piece_assemble.descriptor import DescriptorExtractor, DummyDescriptorExtractor
from piece_assemble.piece import Piece, TransformedPiece

if TYPE_CHECKING:
    from piece_assemble.types import BinImg, NpImage


def load_images(
    img_dir: str, scale: float = 1
) -> tuple[list[str], list[NpImage], list[BinImg]]:
    """Load all piece images from the given directory.

    Parameters
    ----------
    img_dir
        Path to the directory containing piece images.
        Two images are required for every image, one PNG binary mask with "_mask" suffix
        and one JPG image with piece itself, both need to have the same size.
    scale
        All returned images will be rescaled by this factor.

    Returns
    -------
    img_ids
        A list of image names.
    imgs
        A list of 3-channel images of pieces
    masks
        A list of binary piece masks.
    """
    img_ids = [
        name
        for name in os.listdir(img_dir)
        if not name.endswith("_mask.png")
        and name not in ["pieces.json", "original.png"]
    ]

    imgs = [load_img(os.path.join(img_dir, name), scale) for name in img_ids]
    img_ids = [name.split(".")[0] for name in img_ids]
    masks = [
        load_bin_img(os.path.join(img_dir, f"{name}_mask.png"), scale)
        for name in img_ids
    ]
    return img_ids, imgs, masks


def load_pieces(
    path: str, descriptor: DescriptorExtractor | None = None
) -> dict[Piece]:
    """
    Load pieces from the given directory.

    Parameters
    ----------
    path
        Path to the directory containing piece images.

    Returns
    -------
    pieces
        A dictionary of Piece objects.
    """
    if descriptor is None:
        descriptor = DummyDescriptorExtractor()

    img_ids, imgs, masks = load_images(path)

    return {
        img_ids[i]: Piece.from_image(img_ids[i], imgs[i], masks[i], descriptor, 0)
        for i in range(len(img_ids))
    }


def load_puzzle(path: str) -> tuple[dict[TransformedPiece], list[list[str]]]:
    """
    Load puzzle from the given directory.

    Puzzle is represented as a dictionary of TransformedPiece objects and
    a list of lists of neighbor piece names.

    Parameters
    ----------
    path
        Path to the directory containing puzzle pieces.

    Returns
    -------
    pieces
        A dictionary of TransformedPiece objects.
    neighbors
        A list of lists of neighbor piece names.
    """
    pieces = load_pieces(path)

    with open(os.path.join(path, "pieces.json"), "r") as f:
        pieces_json = json.load(f)

    transformed_pieces = {
        p["id"]: TransformedPiece(
            pieces[p["id"]], Transformation.from_dict(p["transformation"])
        )
        for p in pieces_json["transformed_pieces"]
    }

    return transformed_pieces, pieces_json["neighbors"]
