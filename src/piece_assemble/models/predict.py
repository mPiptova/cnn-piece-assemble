from itertools import combinations
from typing import Any, Mapping

import numpy as np
import torch
from skimage.transform import hough_line, hough_line_peaks
from torch import nn

from piece_assemble.dataset import (
    BatchCollator,
    get_img_patches_from_piece,
    preprocess_piece_data,
)
from piece_assemble.geometry import draw_line_polar
from piece_assemble.matching.match import CandidateMatch, Match
from piece_assemble.models import EmbeddingUnet, PairNetwork
from piece_assemble.piece import Piece
from piece_assemble.types import NpImage


def model_output_to_candidate_match(
    piece1: Piece,
    piece2: Piece,
    output: np.ndarray,
    threshold: float,
) -> CandidateMatch | None:
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
    tested_angles = np.linspace(np.pi / 6, np.pi / 3, 30, endpoint=False)
    h, theta, d = hough_line(img, theta=tested_angles)

    accum, angles, dists = hough_line_peaks(h, theta, d)
    if accum[0] < min_strength:
        return None

    return (dists[0], angles[0])


def detect_lines(img: NpImage) -> tuple[np.ndarray, np.ndarray]:
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


def embeddings_to_correspondence_matrix(
    embedding_first: np.ndarray, embedding_second: np.ndarray
) -> np.ndarray:
    output = embedding_first.transpose(1, 0) @ embedding_second
    output = nn.functional.sigmoid(torch.from_numpy(output)).numpy()
    # Flip one of the axis so the indexes correspond to the original contour indexes
    output = output[:, ::-1]

    return output


def get_matches(
    model: PairNetwork,
    pieces: dict[str, Piece],
    activation_threshold: float,
) -> list[CandidateMatch]:

    embeddings_first, embeddings_second = compute_piece_embeddings(model, pieces)

    all_pairs = set([tuple(sorted((x, y))) for x, y in list(combinations(pieces, 2))])
    matches = []

    for p1, p2 in all_pairs:
        embedding1 = embeddings_first[p1]
        embedding2 = embeddings_second[p2]

        output = embeddings_to_correspondence_matrix(embedding1, embedding2)

        piece1 = pieces[p1]
        piece2 = pieces[p2]
        piece1 = piece1.transform(piece2.transformation.inverse())
        piece2 = piece2.transform(piece2.transformation.inverse())

        match = model_output_to_candidate_match(
            piece1, piece2, output, activation_threshold
        )

        if match is not None:
            matches.append(match)

    return matches
