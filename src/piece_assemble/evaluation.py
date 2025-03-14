from __future__ import annotations

from typing import TYPE_CHECKING

from piece_assemble.geometry import Transformation

if TYPE_CHECKING:
    PieceTransformations = dict[str, Transformation]
    from piece_assemble.cluster import Cluster


def cluster_to_transformations(cluster: Cluster) -> PieceTransformations:
    return {piece.name: piece.transformation for piece in cluster.pieces.values()}


def cluster_dict_to_transformations(cluster_dict: dict) -> PieceTransformations:
    """Converts a cluster dictionary to a dictionary of transformations.

    Parameters
    ----------
    cluster_dict
        Dictionary containing the definition of a cluster

    Returns
    -------
    piece_transformations
        Dictionary mapping piece ids to transformations
    """
    return {
        piece_id: Transformation.from_dict(transformation)
        for piece_id, transformation in cluster_dict["transformed_pieces"].items()
    }


def _fixed_position_correct_piece_ratio(
    predicted: PieceTransformations,
    ground_truth: PieceTransformations,
    angle_tol: float,
    translation_tol: float,
) -> float:
    """
    Calculate the ratio of correctly transformed pieces between two assemblies.

    Both dictionaries should have the same keys (piece ids). The values of the
    dictionaries are the transformations of the pieces. The function will consider
    the transformations as correct if the angle difference between the two
    transformations is less than `angle_tol` and the difference in translation is
    less than `translation_tol`.

    Parameters
    ----------
    pred_transformations
        The predicted transformations.
    true_transformations
        The true transformations.
    angle_tol
        The maximum angle difference between the two transformations.
    translation_tol
        The maximum difference in translation between the two transformations.

    Returns
    -------
    ratio
        The ratio of correctly transformed pieces.
    """

    # Corner cases
    if len(ground_truth) == 0:
        return 1
    if len(predicted) == 0:
        return 0

    correct_number = 0
    for piece_id, true_transformation in ground_truth.items():
        pred_transformation = predicted.get(piece_id, None)
        if pred_transformation is None:
            continue
        if pred_transformation.is_close(
            true_transformation, angle_tol, translation_tol
        ):
            correct_number += 1

    return correct_number / len(ground_truth)


def unify_transformations(
    predicted: PieceTransformations, ground_truth: PieceTransformations, key: str
) -> PieceTransformations | None:
    """Transform the predicted pieces to match the transformation of given piece.

    Parameters
    ----------
    predicted
        Representation of the predicted pieces transformations.
    ground_truth
        Representation of the ground truth pieces transformations.
    key
        The piece whose transformation will be used to unify the transformations.

    Returns
    -------
    unified_transformations
        Predicted transformations rotated and translated in such a way that the
        transformation of piece `key` is the same as the ground truth. Other pieces are
        transformed correspondingly.

    """
    true_t = ground_truth.get(key, None)
    pred_t = predicted.get(key, None)
    if pred_t is None or true_t is None:
        return None

    unifying_transformation = pred_t.inverse().compose(true_t)

    return {
        piece_id: t.compose(unifying_transformation)
        for piece_id, t in predicted.items()
    }


def correct_piece_ratio(
    predicted: PieceTransformations,
    ground_truth: PieceTransformations,
    angle_tol: float,
    translation_tol: float,
) -> float:
    """
    Calculate the ratio of correctly transformed pieces, given two assemblies.

    Parameters
    ----------
    predicted
        Representation of the predicted pieces transformations.
    ground_truth
        Representation of the ground truth pieces transformations.
    angle_tol
        The maximum angle difference between the predicted and ground truth
        transformations for them to be considered the same
    translation_tol
        The maximum distance difference between the predicted and ground truth
        transformations for them to be considered the same

    Returns
    -------
    ratio
        The ratio of correctly transformed pieces
    """
    max_ratio = 0
    for piece_id in ground_truth.keys():
        unified_pred = unify_transformations(predicted, ground_truth, piece_id)
        if unified_pred is None:
            continue

        ratio = _fixed_position_correct_piece_ratio(
            unified_pred, ground_truth, angle_tol, translation_tol
        )
        if ratio > max_ratio:
            max_ratio = ratio

    return max_ratio


def _merge_overlapping_clusters(clusters: list[set[str]]) -> list[set[str]]:
    something_changed = True
    while something_changed:
        something_changed = False
        for i, cluster in enumerate(clusters):
            for j in range(i + 1, len(clusters)):
                other_cluster = clusters[j]
                if cluster.intersection(other_cluster):
                    clusters.pop(j)
                    clusters.pop(i)
                    clusters.append(cluster.union(other_cluster))
                    something_changed = True
    return clusters


def _get_correct_clusters(
    assembled: PieceTransformations,
    ground_truth: PieceTransformations,
    angle_tol: float,
    translation_tol: float,
) -> list[set[str]]:
    if len(assembled) == 0:
        return []

    clusters: list[set[str]] = []

    for piece_id in assembled.keys():
        unified_pred = unify_transformations(assembled, ground_truth, piece_id)
        if unified_pred is None:
            continue
        clusters.append(
            {
                p_id
                for p_id in unified_pred.keys()
                if unified_pred[p_id].is_close(
                    ground_truth[p_id], angle_tol, translation_tol
                )
            }
        )
        clusters = _merge_overlapping_clusters(clusters)

    return clusters


def get_correctly_predicted_clusters(
    predicted: list[PieceTransformations],
    ground_truth: PieceTransformations,
    angle_tol: float,
    translation_tol: float,
) -> list[PieceTransformations]:
    largest_cluster_per_piece = {
        id: {id: Transformation.identity()} for id in ground_truth.keys()
    }
    for pred_cluster in predicted:
        correct_clusters_pieces = _get_correct_clusters(
            pred_cluster, ground_truth, angle_tol, translation_tol
        )
        correct_clusters = [
            {piece_id: pred_cluster[piece_id] for piece_id in cluster_pieces}
            for cluster_pieces in correct_clusters_pieces
        ]
        for cluster in correct_clusters:
            for piece_id in cluster.keys():
                if len(largest_cluster_per_piece[piece_id]) < len(cluster):
                    largest_cluster_per_piece[piece_id] = cluster.copy()

    largest_clusters = sorted(
        list(largest_cluster_per_piece.values()),
        key=lambda cluster: len(cluster),
        reverse=True,
    )

    unique_clusters = []
    while len(largest_clusters) > 0:
        new_unique_cluster = largest_clusters.pop(0)
        unique_clusters.append(new_unique_cluster)
        for c in largest_clusters:
            for key in new_unique_cluster.keys():
                c.pop(key, None)

        largest_clusters = [cluster for cluster in largest_clusters if len(cluster) > 0]

    return unique_clusters
