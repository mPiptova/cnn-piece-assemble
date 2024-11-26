from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Generator, Sequence

import numpy as np
import pandas as pd
import torch

from piece_assemble.models.data import img_to_patches

if TYPE_CHECKING:
    from piece_assemble.piece import Piece
    from piece_assemble.types import Points


class PairsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        circular_padding: int = 4,
        seed: int = 42,
        batch_size: int = 8,
        negative_ratio: float = 0.1,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.circular_padding = circular_padding
        self.batch_size = batch_size
        self.random = np.random.default_rng(seed)

        self.pieces_data_index = pd.read_csv(
            os.path.join(dataset_dir, "data_index.csv"), header=None, dtype=str
        )

        self.neighbors_index = pd.read_csv(
            os.path.join(dataset_dir, "neighbors_index.csv"), header=None, dtype=str
        )

        self.negative_pairs = list(
            self.generate_negative_pairs(
                int(len(self.neighbors_index) * negative_ratio)
            )
        )

    def generate_negative_pairs(
        self, count: int
    ) -> Generator[tuple[str, str], None, None]:
        """Generate negative pairs.

        Parameters
        ----------
        count
            Number of negative pairs to generate.
        """
        positive_pairs = [
            (pair[0], pair[1]) for _, pair in self.neighbors_index.iterrows()
        ]

        counter = 0
        while counter < count:
            piece_pair = tuple(
                self.random.choice(self.pieces_data_index[0], 2, replace=False)
            )
            piece_pair2 = (piece_pair[1], piece_pair[0])

            if piece_pair not in positive_pairs and piece_pair2 not in positive_pairs:
                counter += 1
                yield piece_pair

    def __len__(self) -> int:
        return len(self.neighbors_index) + len(self.negative_pairs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if idx < len(self.neighbors_index):
            row = self.neighbors_index.iloc[idx]
            piece1 = row[0]
            piece2 = row[1]
            neighbor_file = row[2]
        else:
            piece1, piece2 = self.negative_pairs[idx - len(self.neighbors_index)]
            neighbor_file = None

        if neighbor_file is not None:
            with np.load(os.path.join(self.dataset_dir, neighbor_file)) as data:
                matrix = data[f"{piece1}-{piece2}"]

            # in this case, piece1 and piece2 are located in the same file
            data_file = self.pieces_data_index.loc[
                self.pieces_data_index[0] == piece1, 1
            ].item()

            with np.load(os.path.join(self.dataset_dir, data_file)) as data:
                piece1_data = preprocess_piece_data(data[piece1])
                piece2_data = preprocess_piece_data(data[piece2])

        else:
            matrix = None
            data_file1 = self.pieces_data_index.loc[
                self.pieces_data_index[0] == piece1, 1
            ].item()
            data_file2 = self.pieces_data_index.loc[
                self.pieces_data_index[0] == piece2, 1
            ].item()

            with np.load(os.path.join(self.dataset_dir, data_file1)) as data:
                piece1_data = data[piece1]

            with np.load(os.path.join(self.dataset_dir, data_file2)) as data:
                piece2_data = data[piece2]

        return piece1_data, piece2_data, matrix


def preprocess_piece_data(data: np.ndarray) -> np.ndarray:
    return 1 - data * 2


class BatchCollator:
    def __init__(self, padding: int, len_divisor: int = 8):
        self.padding = padding
        self.len_divisor = len_divisor

    def __call__(
        self, batch: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        pieces1, pieces2, matrices = zip(*batch)
        return self._prepare_batch_data(pieces1, pieces2, matrices)

    def _data_to_tensor(
        self, data: np.ndarray, max_length: int | None = None, flip: bool = False
    ) -> torch.Tensor:
        tensor = torch.tensor(data).type(torch.float32)
        # switch channels and dimensions
        tensor = tensor.T

        if flip:
            tensor = torch.flip(tensor, [1])

        # add padding
        tensor = torch.nn.functional.pad(
            tensor, (self.padding, self.padding), mode="circular"
        )

        data_size = tensor.shape[1]

        if max_length is None:
            max_length = data_size
        tensor = torch.nn.functional.pad(
            tensor, (0, max_length - data_size), mode="constant", value=0
        )
        # tensor = tensor.zeros_like(tensor)
        return tensor, torch.tensor(data_size - 2 * self.padding)

    def _get_max_piece_size(self, pieces: list[np.ndarray]) -> int:
        max_size = max([piece.shape[0] for piece in pieces])
        max_size = max_size + 2 * self.padding
        if max_size % self.len_divisor != 0:
            max_size = math.ceil(max_size / self.len_divisor) * self.len_divisor
        return max_size  # type: ignore

    def _prepare_batch_data(
        self,
        pieces1: Sequence[np.ndarray],
        pieces2: Sequence[np.ndarray],
        matrices: Sequence[np.ndarray | None],
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        max_piece_size = self._get_max_piece_size(pieces1 + pieces2)

        pieces1, sizes1 = zip(
            *[self._data_to_tensor(piece, max_piece_size) for piece in pieces1]
        )
        pieces1 = torch.stack(pieces1)
        sizes1 = torch.stack(sizes1)

        pieces2, sizes2 = zip(
            *[
                self._data_to_tensor(piece, max_piece_size, flip=True)
                for piece in pieces2
            ]
        )
        pieces2 = torch.stack(pieces2)
        sizes2 = torch.stack(sizes2)

        def prepare_matrix(
            matrix: np.ndarray | None, size1: int, size2: int, max_piece_size: int
        ) -> torch.Tensor:
            if matrix is None:
                matrix = torch.zeros((size1, size2), dtype=torch.float32)
            else:
                matrix = torch.tensor(matrix).type(torch.float32)
                matrix = torch.flip(matrix, [1])

            padded_matrix = torch.full(
                (max_piece_size, max_piece_size), fill_value=-1.0, dtype=torch.float32
            )
            padded_matrix[:size1, :size2] = matrix
            return padded_matrix

        matrices = [
            prepare_matrix(matrix, size1, size2, max_piece_size - 2 * self.padding)
            for matrix, size1, size2 in zip(matrices, sizes1, sizes2)
        ]

        matrices = torch.stack(matrices)

        return (pieces1, pieces2), matrices


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
    patches_array = np.array(patches)
    return patches_array.reshape((patches_array.shape[0], -1))
