import math
from typing import Mapping

from rustworkx import PyGraph, connected_components

from piece_assemble.models.predict import Match, Predictor
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
    predictor: Predictor,
    pieces: Mapping[str, TransformedPiece],
    neighbors: list[list[str]],
    recall_only: bool = False,
) -> dict:

    neighbors_set = set([tuple(sorted((x, y))) for x, y in neighbors])

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    missed = 0
    wrong = 0
    extra = 0

    pairs = neighbors if recall_only else None
    matches = predictor.predict_matches(pieces, all_pairs=pairs)
    matches = [match.verify(5) for match in matches]

    pred_neighbors_set = set()

    tp_matches = []
    for match in matches:
        if match is None:
            continue

        pred_neighbors_set.add(tuple(sorted((match.id1, match.id2))))

        piece1 = pieces[match.id1]
        piece2 = pieces[match.id2]

        piece1 = piece1.transform(piece2.transformation.inverse())

        if tuple(sorted((match.id1, match.id2))) in neighbors_set:
            if match.transformation.is_close(piece1.transformation):
                tp += 1
                tp_matches.append(match)
            else:
                wrong += 1
                fn += 1

        else:
            fp += 1
            extra += 1

    missed = len(neighbors_set.difference(pred_neighbors_set))
    fn += missed
    tn = math.comb(len(pieces), 2) - fp - fn - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    lcc = eval_assembly_potential(tp_matches, len(pieces))

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
    predictor: Predictor,
    puzzles: list[tuple[dict[str, TransformedPiece], list[list[str]]]],
    recall_only: bool = False,
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
        metrics = eval_puzzle(predictor, pieces, neighbors, recall_only)
        for k, v in metrics.items():
            aggr_metrics[k] += v

    aggr_metrics["precision"] /= len(puzzles)
    aggr_metrics["recall"] /= len(puzzles)
    aggr_metrics["f1"] /= len(puzzles)
    aggr_metrics["accuracy"] /= len(puzzles)
    aggr_metrics["lcc"] /= len(puzzles)
    aggr_metrics["fa"] /= len(puzzles)

    return aggr_metrics
