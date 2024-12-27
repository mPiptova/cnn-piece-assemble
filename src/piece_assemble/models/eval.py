from itertools import combinations

from torch import nn
from tqdm import tqdm

from piece_assemble.models import PairNetwork
from piece_assemble.models.predict import (
    compute_piece_embeddings,
    embeddings_to_correspondence_matrix,
    model_output_to_compact_match,
)
from piece_assemble.piece import TransformedPiece


def eval_puzzle(
    model: PairNetwork,
    pieces: dict[str, TransformedPiece],
    neighbors: list[list[str]],
    window_size: int,
    activation_threshold: float = 0.8,
) -> dict:
    model.eval()

    embeddings1, embeddings2 = compute_piece_embeddings(model, pieces, window_size)

    neighbors_set = set([tuple(sorted((x, y))) for x, y in neighbors])
    all_pairs = set([tuple(sorted((x, y))) for x, y in list(combinations(pieces, 2))])

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    missed = 0
    wrong = 0
    extra = 0

    for p1, p2 in tqdm(all_pairs, desc="Evaluating puzzle"):
        embedding1 = embeddings1[p1]
        embedding2 = embeddings2[p2]
        output = embeddings_to_correspondence_matrix(embedding1, embedding2)

        piece1 = pieces[p1]
        piece2 = pieces[p2]

        piece1 = piece1.transform(piece2.transformation.inverse())
        piece2 = piece2.transform(piece2.transformation.inverse())

        match = model_output_to_compact_match(
            piece1.to_piece(), piece2.to_piece(), output, activation_threshold, 5
        )
        if match is None:
            if tuple(sorted((p1, p2))) in neighbors_set:
                missed += 1
                fn += 1
            else:
                tn += 1
            continue

        if tuple(sorted((p1, p2))) in neighbors_set:
            if match.transformation.is_close(piece1.transformation):
                tp += 1
            else:
                wrong += 1
                fn += 1

        else:
            fp += 1
            extra += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed": missed,
        "wrong": wrong,
        "extra": extra,
    }


def eval_puzzles(
    model: nn.Module,
    puzzles: list[tuple[dict[str, TransformedPiece], list[list[str]]]],
    window_size: int,
    activation_threshold: float = 0.8,
) -> dict[str, float]:
    aggr_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0,
        "missed": 0,
        "wrong": 0,
        "extra": 0,
    }

    for pieces, neighbors in puzzles:
        metrics = eval_puzzle(
            model, pieces, neighbors, window_size, activation_threshold
        )
        for k, v in metrics.items():
            aggr_metrics[k] += v

    aggr_metrics["precision"] /= len(puzzles)
    aggr_metrics["recall"] /= len(puzzles)
    aggr_metrics["f1"] /= len(puzzles)

    return aggr_metrics
