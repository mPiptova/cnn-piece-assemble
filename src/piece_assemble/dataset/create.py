"""
Script and tools for creating dataset for training the model.

Example usage:

python src/piece_assemble/dataset/create.py \
    --window-size 7 output/dir puzzle/dir1 puzzle/dir2
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from piece_assemble.dataset import get_img_patches_from_piece
from piece_assemble.load import load_puzzle
from piece_assemble.models.data import get_correspondence_matrix
from piece_assemble.piece import TransformedPiece


def rename_pieces(
    pieces: dict[str, TransformedPiece],
    neighbors: list[list[str]],
    offset: int,
    id_length: int = 7,
) -> tuple[dict[str, TransformedPiece], list[list[str]]]:
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
        piece.name: f"{i + offset:0{id_length}}"
        for i, piece in enumerate(pieces.values())
    }

    renamed_pieces = {id_mappings[piece.name]: piece for piece in pieces.values()}
    for piece in renamed_pieces.values():
        piece.name = id_mappings[piece.name]

    renamed_neighbors = [
        [id_mappings[piece] for piece in neighbor] for neighbor in neighbors
    ]

    return renamed_pieces, renamed_neighbors


def create_dataset(
    puzzle_dirs: list[str],
    target_dir: str,
    window_size: int = 7,
    background_val: float = 1,
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
        pieces, neighbors = load_puzzle(puzzle_dir, background_val)
        pieces, neighbors = rename_pieces(pieces, neighbors, offset)
        offset += len(pieces)

        store_neighbors(pieces, neighbors, target_dir, neighbors_index_path, i)
        store_data(pieces, window_size, target_dir, data_index_path, i)


def store_data(
    pieces: dict[str, TransformedPiece],
    window_size: int,
    target_dir: str,
    data_index_path: str,
    i: int,
) -> None:
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
            cols = [piece.name, piece_data_name]
            f.write(f"{','.join(cols)}\n")

    np.savez_compressed(
        piece_data_path,
        **{
            piece.name: get_img_patches_from_piece(piece.to_piece(), window_size)
            for piece in pieces.values()
        },
    )


def store_neighbors(
    pieces: dict[str, TransformedPiece],
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
    parser.add_argument("--background-val", type=float, default=1)
    args = parser.parse_args()

    # Create dataset dir if it doesn't exist
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    create_dataset(
        args.puzzle_dirs, args.target_dir, args.window_size, args.background_val
    )
