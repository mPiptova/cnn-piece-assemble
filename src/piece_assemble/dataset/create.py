"""
Script and tools for creating dataset for training the model.

Example usage:

python src/piece_assemble/dataset/create.py \
    --window-size 7 output/dir puzzle/dir1 puzzle/dir2
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from piece_assemble.piece import Piece
    from piece_assemble.types import Points

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from geometry import Transformation
from piece_assemble.descriptor import DummyDescriptorExtractor
from piece_assemble.models.data import get_correspondence_matrix, img_to_patches
from piece_assemble.piece import Piece, TransformedPiece
from piece_assemble.tools.run import load_images


def load_pieces(path: str) -> dict[Piece]:
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
    img_ids, imgs, masks = load_images(path)

    return {
        img_ids[i]: Piece(img_ids[i], imgs[i], masks[i], DummyDescriptorExtractor(), 0)
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


def rename_pieces(
    pieces: dict[TransformedPiece],
    neighbors: list[list[str]],
    offset: int,
    id_length: int = 7,
) -> tuple[dict[TransformedPiece], list[list[str]]]:
    """
    Rename pieces in the given dictionary.

    New IDs are numeric values starting from given offset,
    represented as string and padded with leading zeros.

    Parameters
    ----------
    pieces
        A dictionary of TransformedPiece objects.
    neighbors
        A list of lists of neighbor piece names.
    offset
        Offset to add to the piece IDs.
    id_length
        Length of the piece IDs.

    Returns
    -------
    pieces
        A dictionary of TransformedPiece objects.
    neighbors
        A list of lists of neighbor piece names.
    """
    id_mappings = {
        piece.piece.name: f"{i + offset:0{id_length}}"
        for i, piece in enumerate(pieces.values())
    }

    renamed_pieces = {
        id_mappings[piece.piece.name]: TransformedPiece(
            piece.piece, piece.transformation
        )
        for piece in pieces.values()
    }
    for piece in renamed_pieces.values():
        piece.piece.name = id_mappings[piece.piece.name]

    renamed_neighbors = [
        [id_mappings[piece] for piece in neighbor] for neighbor in neighbors
    ]

    return renamed_pieces, renamed_neighbors


def create_dataset(
    puzzle_dirs: list[str], target_dir: str, window_size: int = 7
) -> None:
    """
    Create and store dataset from the given puzzle directories.

    Parameters
    ----------
    puzzle_dirs
        A list of paths to the directories containing puzzle pieces.
    target_dir
        Path to the directory where the dataset will be stored.
    window_size
        Size of the sliding window used to extract patches.
    """
    # Create target dir if it doesn't exist
    neighbors_index_path = os.path.join(target_dir, "neighbors_index.csv")
    data_index_path = os.path.join(target_dir, "data_index.csv")

    # Reset indices if they already exist
    if os.path.exists(neighbors_index_path):
        os.remove(neighbors_index_path)
    if os.path.exists(data_index_path):
        os.remove(data_index_path)

    offset = 0
    for i, puzzle_dir in tqdm(enumerate(puzzle_dirs)):
        pieces, neighbors = load_puzzle(puzzle_dir)
        pieces, neighbors = rename_pieces(pieces, neighbors, offset)
        offset += len(pieces)

        store_neighbors(pieces, neighbors, target_dir, neighbors_index_path, i)
        store_data(pieces, window_size, target_dir, data_index_path, i)


def get_img_patches_from_piece(piece: Piece, window_size: int) -> np.ndarray:
    """
    Extract image patches from the given piece.

    Parameters
    ----------
    piece
        A Piece object.
    window_size
        Size of the sliding window used to extract patches.

    Returns
    -------
    patches
        An array of image patches.
    """
    return get_img_patches(piece.contour, piece.img, window_size)


def get_img_patches(contour: Points, img: np.ndarray, window_size: int) -> np.ndarray:
    patches = img_to_patches(contour, img, window_size)
    patches = np.array(patches)
    return patches.reshape((patches.shape[0], -1))


def store_data(
    pieces: dict[TransformedPiece],
    window_size: int,
    target_dir: str,
    data_index_path: str,
    i: int,
):
    """
    Store image patches from the given pieces.

    Parameters
    ----------
    pieces
        A dictionary of TransformedPiece objects.
    window_size
        Size of the sliding window used to extract patches.
    target_dir
        Path to the directory where the dataset will be stored.
    data_index_path
        Path to the data index file.
    i
        Index of the puzzle.
    """
    piece_data_name = f"data_{i}.npz"
    piece_data_path = os.path.join(target_dir, piece_data_name)

    with open(data_index_path, "a+") as f:
        for piece in pieces.values():
            cols = [piece.piece.name, piece_data_name]
            f.write(f"{','.join(cols)}\n")

    np.savez_compressed(
        piece_data_path,
        **{
            piece.piece.name: get_img_patches_from_piece(piece.piece, window_size)
            for piece in pieces.values()
        },
    )


def store_neighbors(
    pieces: dict[TransformedPiece],
    neighbors: list[list[str]],
    target_dir: str,
    neighbors_index_path: str,
    i: int,
) -> None:
    """
    Store neighbor matrices from the given pieces.

    Parameters
    ----------
    pieces
        A dictionary of TransformedPiece objects.
    neighbors
        A list of lists of neighbor piece names.
    target_dir
        Path to the directory where the dataset will be stored.
    neighbors_index_path
        Path to the neighbors index file.
    i
        Index of the puzzle.
    """
    neighbor_matrices_name = f"neighbors_{i}.npz"
    neighbor_matrices_path = os.path.join(target_dir, neighbor_matrices_name)

    with open(neighbors_index_path, "a+") as f:
        for neighbor in neighbors:
            cols = list(neighbor) + [neighbor_matrices_name]
            f.write(f"{','.join(cols)}\n")

    np.savez_compressed(
        neighbor_matrices_path,
        **{
            f"{pair[0]}-{pair[1]}": get_correspondence_matrix(
                pieces[pair[0]], pieces[pair[1]]
            )
            for pair in neighbors
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str)
    parser.add_argument("puzzle_dirs", type=str, nargs="+")
    parser.add_argument("--window-size", type=int, default=7)
    args = parser.parse_args()

    # Create dataset dir if it doesn't exist
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    create_dataset(args.puzzle_dirs, args.target_dir, args.window_size)
