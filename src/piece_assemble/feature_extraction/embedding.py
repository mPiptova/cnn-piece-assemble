from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from piece_assemble.dataset import BatchCollator, get_img_patches, preprocess_piece_data
from piece_assemble.feature_extraction.base import FeatureExtractor, Features
from piece_assemble.matching.match import Match
from piece_assemble.models import PairNetwork
from piece_assemble.models.predict import (
    embeddings_to_correspondence_matrix,
    model_output_to_match,
)

if TYPE_CHECKING:
    from piece_assemble.piece import Piece
    from piece_assemble.types import BinImg, NpImage, Points


class EmbeddingFeatures(Features):
    def __init__(self, emb1: np.ndarray, emb2: np.ndarray):
        super().__init__()
        self.emb1 = emb1
        self.emb2 = emb2

    def get_complexity(self, idxs: np.ndarray) -> float:
        return 1


class EmbeddingExtractor(FeatureExtractor):
    def __init__(
        self, model: PairNetwork, patch_size: int = 7, activation_threshold: float = 0.8
    ):
        self.model = model
        self.collator = BatchCollator(model.padding)
        self.activation_threshold = activation_threshold
        self.patch_size = patch_size

    def extract(self, contour: Points, image: NpImage) -> EmbeddingFeatures:
        data = get_img_patches(contour, image, self.patch_size)
        data = preprocess_piece_data(data)
        input, _ = self.collator([(data, data, None)])
        device = next(self.model.parameters()).device
        input = (input[0].to(device), input[1].to(device))

        self.model.eval()
        with torch.no_grad():
            e_first = (
                self.model.embedding_network1(input[0])
                .detach()
                .cpu()
                .numpy()[0][:, : data.shape[0]]
            )
            e_second = (
                self.model.embedding_network2(input[1])
                .detach()
                .cpu()
                .numpy()[0][:, : data.shape[0]]
            )

        return EmbeddingFeatures(e_first, e_second)

    def dist(self, first: Piece, second: Piece) -> np.ndarray:
        if (
            type(first.features) != EmbeddingFeatures
            or type(second.features) != EmbeddingFeatures
        ):
            raise ValueError("Features must be of type EmbeddingFeatures")
        return 1 - embeddings_to_correspondence_matrix(
            first.features.emb1, second.features.emb2
        )

    def prepare_image(
        self, image: NpImage, mask: BinImg, blur_image: NpImage
    ) -> NpImage:
        image = image.copy()
        image[~mask] = self.model.background_val
        return image

    def find_all_matches(
        self,
        pieces: list[Piece],
    ) -> list[Match]:
        all_pairs = [(x, y) for x, y in list(combinations(pieces, 2))]
        matches = []

        for piece1, piece2 in tqdm(all_pairs, desc="Finding matches"):
            output = embeddings_to_correspondence_matrix(
                piece1.features.emb1, piece2.features.emb2
            )
            match = model_output_to_match(
                piece1, piece2, output, self.activation_threshold
            )
            if match is not None:
                matches.append(match)
        return matches
