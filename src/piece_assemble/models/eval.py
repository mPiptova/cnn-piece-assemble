from itertools import combinations
from typing import Mapping

from rustworkx import PyGraph, connected_components
from torch import nn
from tqdm import tqdm

from piece_assemble.models import PairNetwork
from piece_assemble.models.predict import (
    Match,
    compute_piece_embeddings,
    embeddings_to_correspondence_matrix,
    model_output_to_match,
)
from piece_assemble.piece import TransformedPiece


def eval_assembly_potential(matches: list[Match], n_pieces: int) -> float:
    if len(matches) == 0:
        return 0
    graph = PyGraph()
    id_mapping = {}
    for match in matches:
        for piece_id in [match.id1, match.id2]:
            if not id_mapping.get(piece_id, False):
                i = len(id_mapping)
                id_mapping[piece_id] = i
                graph.add_node(i)
        graph.add_edge(id_mapping[match.id1], id_mapping[match.id2], len(graph.edges()))

    components = connected_components(graph)
    return max([len(component) for component in components]) / n_pieces


def eval_puzzle(
    model: PairNetwork,
    pieces: Mapping[str, TransformedPiece],
    neighbors: list[list[str]],
    activation_threshold: float = 0.8,
) -> dict:
    model.eval()

    embeddings1, embeddings2 = compute_piece_embeddings(model, pieces)

    neighbors_set = set([tuple(sorted((x, y))) for x, y in neighbors])
    all_pairs = set([tuple(sorted((x, y))) for x, y in list(combinations(pieces, 2))])

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    missed = 0
    wrong = 0
    extra = 0

    true_matches = []

    for p1, p2 in tqdm(all_pairs, desc="Evaluating puzzle"):
        embedding1 = embeddings1[p1]
        embedding2 = embeddings2[p2]
        output = embeddings_to_correspondence_matrix(embedding1, embedding2)

        piece1 = pieces[p1]
        piece2 = pieces[p2]

        piece1 = piece1.transform(piece2.transformation.inverse())
        piece2 = piece2.transform(piece2.transformation.inverse())

        match = model_output_to_match(
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
                true_matches.append(match)
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
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    lcc = eval_assembly_potential(true_matches, len(pieces))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed": missed,
        "wrong": wrong,
        "extra": extra,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "lcc": lcc,
        "fa": 1 if lcc == 1 else 0,
    }


def eval_puzzles(
    model: nn.Module,
    puzzles: list[tuple[dict[str, TransformedPiece], list[list[str]]]],
    activation_threshold: float = 0.8,
) -> dict[str, float]:
    aggr_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0,
        "missed": 0,
        "wrong": 0,
        "extra": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "accuracy": 0,
        "lcc": 0,
        "fa": 0,
    }

    for pieces, neighbors in puzzles:
        metrics = eval_puzzle(model, pieces, neighbors, activation_threshold)
        for k, v in metrics.items():
            aggr_metrics[k] += v

    aggr_metrics["precision"] /= len(puzzles)
    aggr_metrics["recall"] /= len(puzzles)
    aggr_metrics["f1"] /= len(puzzles)
    aggr_metrics["accuracy"] /= len(puzzles)
    aggr_metrics["lcc"] /= len(puzzles)
    aggr_metrics["fa"] /= len(puzzles)

    return aggr_metrics
