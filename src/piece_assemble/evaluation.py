from piece_assemble.geometry import Transformation


def _fixed_position_correct_piece_ratio(
    pred_transformations: dict,
    true_transformations: dict,
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
    if len(true_transformations) == 0:
        return 1
    if len(pred_transformations) == 0:
        return 0

    correct_number = 0
    for piece_id, true_transformation in true_transformations.items():
        pred_transformation = pred_transformations.get(piece_id, None)
        if pred_transformation is None:
            continue
        if pred_transformation.is_close(
            true_transformation, angle_tol, translation_tol
        ):
            correct_number += 1

    return correct_number / len(true_transformations)


def correct_piece_ratio(
    assembled: dict, ground_truth: dict, angle_tol: float, translation_tol: float
) -> float:
    """
    Calculate the ratio of correctly transformed pieces, given two assemblies.

    Parameters
    ----------
    assembled
        The assembled pieces, as a dictionary with the following keys:
            - transformed_pieces: a list of dictionaries, each containing the keys 'id'
            and 'transformation'
    ground_truth
        The ground truth pieces, as a dictionary with the same structure as `assembled`
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
    true_transformations = {
        piece["id"]: Transformation.from_dict(piece["transformation"])
        for piece in ground_truth["transformed_pieces"]
    }

    pred_transformations = {
        piece["id"]: Transformation.from_dict(piece["transformation"])
        for piece in assembled["transformed_pieces"]
    }

    max_ratio = 0
    for piece_id in true_transformations.keys():
        true_t = true_transformations.get(piece_id, None)
        pred_t = pred_transformations.get(piece_id, None)
        if pred_t is None or true_t is None:
            continue

        true_t = true_t.inverse()
        pred_t = pred_t.inverse()

        normalized_true = {
            piece_id: t.compose(true_t) for piece_id, t in true_transformations.items()
        }
        normalized_pred = {
            piece_id: t.compose(pred_t) for piece_id, t in pred_transformations.items()
        }

        ratio = _fixed_position_correct_piece_ratio(
            normalized_pred, normalized_true, angle_tol, translation_tol
        )
        if ratio > max_ratio:
            max_ratio = ratio

    return max_ratio
