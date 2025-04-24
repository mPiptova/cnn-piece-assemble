from itertools import combinations
from typing import Any, Mapping

import numpy as np
import torch
from skimage.transform import hough_line, hough_line_peaks
from torch import nn
from tqdm import tqdm

from piece_assemble.dataset import (
    BatchCollator,
    get_img_patches_from_piece,
    preprocess_piece_data,
)
from piece_assemble.geometry import draw_line_polar
from piece_assemble.match import CandidateMatch, Match
from piece_assemble.models import EmbeddingUnet, PairNetwork, load_model
from piece_assemble.piece import Piece
from piece_assemble.types import NpImage


class Embeddings:
    """A class to store embeddings for all pieces."""

    def __init__(self, embeddings: tuple[dict[str, np.ndarray], dict[str, np.ndarray]]):
        self._embeddings = embeddings

    def get_similarity_matrix(self, piece_id1: str, piece_id2: str) -> np.ndarray:
        """Get the similarity matrix between two pieces.

        Parameters
        ----------
        piece_id1
            The id of the first piece.
        piece_id2
            The id of the second piece.

        Returns
        -------
        np.ndarray
            The similarity matrix between the two pieces.
        """
        return embeddings_to_similarity_matrix(
            self._embeddings[0][piece_id1], self._embeddings[1][piece_id2]
        )


class EnsembleEmbeddings(Embeddings):
    """A class to store ensemble embeddings for all pieces."""

    def __init__(self, embeddings: list[tuple[dict[str, Any], dict[str, Any]]]):
        self._embeddings = embeddings

    def get_similarity_matrix(self, piece_id1: str, piece_id2: str) -> np.ndarray:
        """Get the similarity matrix between two pieces.

        Parameters
        ----------
        piece_id1
            The id of the first piece.
        piece_id2
            The id of the second piece.

        Returns
        -------
        np.ndarray
            The similarity matrix between the two pieces.
        """
        matrices = [
            embeddings_to_similarity_matrix(
                self._embeddings[i][0][piece_id1], self._embeddings[i][1][piece_id2]
            )
            for i in range(len(self._embeddings))
        ]
        matrix = np.ones_like(matrices[0])
        for m in matrices:
            matrix *= m
        return matrix


class Predictor:
    """A class to predict matches."""

    def __init__(self, model: PairNetwork, activation_threshold: float):
        self.model = model
        self.activation_threshold = activation_threshold

    def predict_matches(
        self,
        pieces: Mapping[str, Piece],
        all_pairs: list[tuple[str, str]] | list[list[str]] | None = None,
    ) -> list[CandidateMatch]:
        """Get candidate matches between all pieces in the puzzle.

        Parameters
        ----------
        pieces
            The pieces in the puzzle.
        all_pairs
            The pairs of pieces to check. If None, all pairs will be checked.
            Can be useful when only recall is needed.

        Returns
        -------
        list[CandidateMatch]
            The candidate matches between all pieces in the puzzle.
        """
        embeddings = self.predict_embeddings(pieces)
        if all_pairs is None:
            all_pairs = set(
                [tuple(sorted((x, y))) for x, y in list(combinations(pieces, 2))]
            )
        matches = []

        for p1, p2 in tqdm(all_pairs, desc="Finding candidate matches"):
            output = embeddings.get_similarity_matrix(p1, p2)

            piece1 = pieces[p1].to_piece()
            piece2 = pieces[p2].to_piece()

            match = model_output_to_candidate_match(
                piece1, piece2, output, self.activation_threshold
            )

            if match is not None:
                matches.append(match)

        return matches

    def predict_embeddings(self, pieces: Mapping[str, Piece]) -> Embeddings:
        """Get embeddings for all pieces in the puzzle."""
        return Embeddings(compute_piece_embeddings(self.model, pieces))


class EnsemblePredictor(Predictor):
    """A class to predict matches using an ensemble of models."""

    def __init__(self, models: list[PairNetwork], activation_threshold: float):
        self.models = models
        self.activation_threshold = activation_threshold

    def predict_embeddings(self, pieces: Mapping[str, Piece]) -> EnsembleEmbeddings:
        """Get embeddings for all pieces in the puzzle.

        Parameters
        ----------
        pieces
            The pieces in the puzzle.
        """

        embeddings = [compute_piece_embeddings(model, pieces) for model in self.models]
        return EnsembleEmbeddings(embeddings)


def load_predictor(
    predictor_id: str, path: str, activation_threshold: float
) -> Predictor:
    """Load a predictor.

    Parameters
    ----------
    predictor_id
        The id of the predictor. Either the ID of a single model, or it has the form
        "Ensemble-model1-model2-model3".
    path
        The path where all models are stored.
    activation_threshold
        The activation threshold for the predictor.

    Returns
    -------
    Predictor
    """

    if predictor_id.startswith("Ensemble"):
        model_ids = predictor_id.split("-")[1:]
        models = [load_model(model_id, path) for model_id in model_ids]
        return EnsemblePredictor(models, activation_threshold)

    model = load_model(predictor_id, path)
    return Predictor(model, activation_threshold)


def model_output_to_candidate_match(
    piece1: Piece,
    piece2: Piece,
    output: np.ndarray,
    threshold: float,
) -> CandidateMatch | None:
    """Converts model output to a CandidateMatch object.

    Parameters
    ----------
    piece1
        The first piece.
    piece2
        The second piece.
    output
        The output of the model.
    threshold
        The threshold for the model output.

    Returns
    -------
    CandidateMatch

    """
    if output.max() < threshold:
        return None

    idxs1, idxs2 = detect_lines(output > threshold)
    if len(idxs1) == 0:
        return None

    match = CandidateMatch(piece1, piece2, idxs1, idxs2, 1 / len(idxs1))

    return match


def model_output_to_match(
    piece1: Piece,
    piece2: Piece,
    output: np.ndarray,
    threshold: float,
    dist_tol: float,
    icp_max_iters: dict | None = None,
    icp_min_change: dict | None = None,
    ios_tol: float | None = None,
) -> Match | None:
    """Converts model output to a Match object.

    Parameters
    ----------
    piece1
        The first piece.
    piece2
        The second piece.
    output
        The output of the model.
    threshold
        The threshold for the model output.
    dist_tol
        The distance tolerance for the match.
    icp_max_iters
        The maximum number of iterations for the ICP algorithm.
    icp_min_change
        The minimum change for the ICP algorithm.
    ios_tol
        The tolerance for the IOS algorithm.

    Returns
    -------
    Match
        The match between the two pieces.
    """
    match = model_output_to_candidate_match(piece1, piece2, output, threshold)
    if match is None:
        return None

    verify_params: dict[str, Any] = {}
    verify_params["dist_tol"] = dist_tol
    if icp_max_iters is not None:
        verify_params["icp_max_iters"] = icp_max_iters
    if icp_min_change is not None:
        verify_params["icp_min_change"] = icp_min_change
    if ios_tol is not None:
        verify_params["ios_tol"] = ios_tol

    return match.verify(**verify_params)


def compute_piece_embeddings(
    model: EmbeddingUnet,
    pieces: Mapping[str, Piece],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute embeddings for all pieces.

    Parameters
    ----------
    model
        EmbeddingUnet model.
    pieces
        A dictionary of Piece objects.

    Returns
    -------
    embeddings_first
        A dictionary of embeddings for each piece. These embeddings are the embeddings
        of the first piece in the pair.
    embeddings_second
        A dictionary of embeddings for each piece. These embeddings are the embeddings
        of the second piece in the pair.

    """
    # Ensure that the model is in evaluation mode
    model.eval()

    collator = BatchCollator(model.padding)

    embeddings_first = {}
    embeddings_second = {}

    for name, piece in pieces.items():
        data = get_img_patches_from_piece(piece.to_piece(), model.window_size)
        p_data = preprocess_piece_data(data)
        input, _ = collator([(p_data, p_data, None)])
        device = next(model.parameters()).device
        input = (input[0].to(device), input[1].to(device))
        with torch.no_grad():
            embeddings_first[name] = (
                model.embedding_network1(input[0])
                .detach()
                .cpu()
                .numpy()[0][:, : data.shape[0]]
            )
            embeddings_second[name] = (
                model.embedding_network2(input[1])
                .detach()
                .cpu()
                .numpy()[0][:, : data.shape[0]]
            )

    return embeddings_first, embeddings_second


def detect_strongest_line(
    img: NpImage, min_strength: int = 60
) -> tuple[float, float] | None:
    """Detect the strongest diagonal line in given image.

    Parameters
    ----------
    img
        The model output.
    min_strength
        The minimum strength of the line.

    Returns
    -------
    The distance and angle of the line (polar coordinates).
    """
    tested_angles = np.linspace(np.pi / 6, np.pi / 3, 30, endpoint=False)
    h, theta, d = hough_line(img, theta=tested_angles)

    accum, angles, dists = hough_line_peaks(h, theta, d)
    if accum[0] < min_strength:
        return None

    return (dists[0], angles[0])


def detect_lines(img: NpImage) -> tuple[np.ndarray, np.ndarray]:
    """Detect the strongest diagonal line in the model output.

    Parameters
    ----------
    img
        The model output.

    Returns
    -------
    The distance and angle of the line (polar coordinates).
    """
    img_tiled = np.tile(img, (2, 2))
    line = detect_strongest_line(img_tiled)
    if line is None:
        return (
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint8),
        )

    dist, angle = line

    img_line = np.zeros_like(img_tiled, dtype=np.uint8)
    draw_line_polar(img_line, (dist, angle), 3)

    img_detected = img_tiled * img_line

    idxs1, idxs2 = np.where(img_detected)
    median1 = np.median(idxs1)
    std1 = np.std(idxs1)
    median2 = np.median(idxs2)
    std2 = np.std(idxs2)

    is_clean = (abs(idxs1 - median1) < std1) & (abs(idxs2 - median2) < std2)
    clean_idxs1 = idxs1[is_clean] % img.shape[0]
    clean_idxs2 = idxs2[is_clean] % img.shape[1]

    return clean_idxs1, clean_idxs2


def embeddings_to_similarity_matrix(
    embedding_first: np.ndarray, embedding_second: np.ndarray
) -> np.ndarray:
    """Convert the embedding output to a similarity matrix.

    Parameters
    ----------
    embeddings_first
        Embeddings of the first piece
    embeddings_second
        Embeddings of the second piece

    Returns
    -------
    The similarity matrix of two pieces.
    """
    output = embedding_first.transpose(1, 0) @ embedding_second
    output = nn.functional.sigmoid(torch.from_numpy(output)).numpy()
    # Flip one of the axis so the indexes correspond to the original contour indexes
    output = output[:, ::-1]

    return output
