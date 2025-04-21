from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from piece_assemble.evaluation import (
    _fixed_position_correct_piece_ratio,
    _get_correct_clusters,
    correct_piece_ratio,
    get_correctly_predicted_clusters,
)
from piece_assemble.geometry import Transformation

if TYPE_CHECKING:
    from piece_assemble.evaluation import PieceTransformations


@pytest.mark.parametrize(
    "pred_transformations, true_transformations, angle_tol, translation_tol, "
    "expected_result",
    [
        ({}, {}, 0, 0, 1),
        ({"0": {"translation": [0, 0], "rotation_angle": 0}}, {}, 0, 0, 1),
        ({}, {"0": {"translation": [0, 0], "rotation_angle": 0}}, 0, 0, 0),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            0,
            0,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, 0], "rotation_angle": 0}},
            0,
            0,
            0,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, 0], "rotation_angle": 0}},
            0,
            10,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 1}},
            0,
            0,
            0,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 1}},
            1,
            0,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, 0], "rotation_angle": 1}},
            1,
            0,
            0,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, 0], "rotation_angle": 1}},
            1,
            10,
            1,
        ),
        (
            {
                "0": {"translation": [0, 0], "rotation_angle": 0},
                "1": {"translation": [10, 0], "rotation_angle": 1},
            },
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            0,
            0,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {
                "0": {"translation": [0, 0], "rotation_angle": 0},
                "1": {"translation": [10, 0], "rotation_angle": 1},
            },
            0,
            0,
            0.5,
        ),
    ],
)
def test_fixed_position_correct_piece_ratio(
    pred_transformations: dict,
    true_transformations: dict,
    angle_tol: float,
    translation_tol: float,
    expected_result: float,
) -> None:
    pred_transformations = {
        id: Transformation.from_dict(t) for id, t in pred_transformations.items()
    }
    true_transformations = {
        id: Transformation.from_dict(t) for id, t in true_transformations.items()
    }
    assert (
        _fixed_position_correct_piece_ratio(
            pred_transformations, true_transformations, angle_tol, translation_tol
        )
        == expected_result
    )


@pytest.mark.parametrize(
    "pred_transformations, true_transformations, angle_tol, translation_tol, "
    "expected_result",
    [
        # Corner cases
        ({}, {}, 0, 0, 0),
        ({"0": {"translation": [0, 0], "rotation_angle": 0}}, {}, 0, 0, 0),
        ({}, {"0": {"translation": [0, 0], "rotation_angle": 0}}, 0, 0, 0),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            0,
            0,
            1,
        ),
        # One piece
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 0.5}},
            0.17,
            21,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, -30], "rotation_angle": 0}},
            0.17,
            21,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, -30], "rotation_angle": 0.5}},
            0.17,
            21,
            1,
        ),
        # Three pieces, all correct
        # Transformation: {'rotation_angle': 0.1, 'translation': [0, 0]},
        (
            {
                "0": {"translation": [5, 0], "rotation_angle": 0.2},
                "1": {"translation": [10, 10], "rotation_angle": 0.3},
                "2": {"translation": [1, 10], "rotation_angle": 0.8},
            },
            {
                "0": {
                    "rotation_angle": 0.3,
                    "translation": [4.975020826390129, -0.4991670832341408],
                },
                "1": {
                    "rotation_angle": 0.4,
                    "translation": [10.94837581924854, 8.951707486311978],
                },
                "2": {
                    "rotation_angle": 0.9,
                    "translation": [1.9933383317463074, 9.850208236133431],
                },
            },
            0.17,
            21,
            1,
        ),
        # Transformation: {'rotation_angle': 0.6, 'translation': [2, 50]},
        (
            {
                "0": {"translation": [5, 0], "rotation_angle": 0.2},
                "1": {"translation": [10, 10], "rotation_angle": 0.3},
                "2": {"translation": [1, 10], "rotation_angle": 0.8},
            },
            {
                "0": {
                    "rotation_angle": 0.8,
                    "translation": [6.1266780745483915, 47.17678763302482],
                },
                "1": {
                    "rotation_angle": 0.8999999999999999,
                    "translation": [15.899780883047136, 52.60693141514643],
                },
                "2": {
                    "rotation_angle": 1.4,
                    "translation": [8.471760348860032, 57.68871367570175],
                },
            },
            0.17,
            21,
            1,
        ),
        # One off
        # Transformation: {'rotation_angle': 0.6, 'translation': [2, 50]},
        (
            {
                "0": {"translation": [5, 0], "rotation_angle": 0.2},
                "1": {"translation": [10, 10], "rotation_angle": 0.3},
                "2": {"translation": [1, 10], "rotation_angle": 0.8},
            },
            {
                "0": {
                    "rotation_angle": 0.8,
                    "translation": [6.1266780745483915, 47.17678763302482],
                },
                "1": {
                    "rotation_angle": 0.8999999999999999,
                    "translation": [15.899780883047136, 52.60693141514643],
                },
                "2": {"rotation_angle": 2, "translation": [0, 0]},
            },
            0.17,
            21,
            0.666,
        ),
    ],
)
def test_correct_piece_ratio(
    pred_transformations: dict,
    true_transformations: dict,
    angle_tol: float,
    translation_tol: float,
    expected_result: float,
) -> None:
    assembled = {
        piece_id: Transformation.from_dict(transformation)
        for piece_id, transformation in pred_transformations.items()
    }
    ground_truth = {
        piece_id: Transformation.from_dict(transformation)
        for piece_id, transformation in true_transformations.items()
    }

    assert correct_piece_ratio(
        assembled, ground_truth, angle_tol, translation_tol
    ) == pytest.approx(expected_result, 0.01)


@pytest.mark.parametrize(
    "pred_transformations, ground_truth, correct_clusters",
    [
        # All are correct
        (
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([10, 10])),
                "2": Transformation(0.8, np.array([1, 10])),
            },
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([10, 10])),
                "2": Transformation(0.8, np.array([1, 10])),
            },
            [{"0", "1", "2"}],
        ),
        # None are correct
        (
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([10, 10])),
                "2": Transformation(0.8, np.array([1, 10])),
            },
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([0, 5])),
                "2": Transformation(0.8, np.array([10, 1])),
            },
            [{"0"}, {"1"}, {"2"}],
        ),
        # Two clusters
        (
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([10, 10])),
                "2": Transformation(0.8, np.array([1, 10])),
                "3": Transformation(0.8, np.array([10, 1])),
            },
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([10, 10])),
                "2": Transformation(0.8, np.array([1, 20])),
                "3": Transformation(0.8, np.array([10, 11])),
            },
            [{"0", "1"}, {"2", "3"}],
        ),
    ],
)
def test_get_correct_clusters(
    pred_transformations: PieceTransformations,
    ground_truth: PieceTransformations,
    correct_clusters: list[set[str]],
) -> None:

    pred_correct_clusters = sorted(
        _get_correct_clusters(pred_transformations, ground_truth, 0.17, 5)
    )
    correct_clusters = sorted(correct_clusters)

    assert pred_correct_clusters == correct_clusters


@pytest.mark.parametrize(
    "pred_transformations, ground_truth, expected_components",
    [
        # All are correct
        (
            [
                {
                    "0": Transformation(0.2, np.array([5, 0])),
                    "1": Transformation(0.3, np.array([10, 10])),
                    "2": Transformation(0.8, np.array([1, 10])),
                }
            ],
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([10, 10])),
                "2": Transformation(0.8, np.array([1, 10])),
            },
            1,
        ),
        # None are correct
        (
            [
                {
                    "0": Transformation(0.2, np.array([5, 0])),
                    "1": Transformation(0.3, np.array([10, 10])),
                    "2": Transformation(0.8, np.array([1, 10])),
                }
            ],
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([0, 5])),
                "2": Transformation(0.8, np.array([10, 1])),
            },
            3,
        ),
        # Two clusters
        (
            [
                {
                    "0": Transformation(0.2, np.array([5, 0])),
                    "1": Transformation(0.3, np.array([10, 10])),
                    "2": Transformation(0.8, np.array([1, 10])),
                    "3": Transformation(0.8, np.array([10, 1])),
                }
            ],
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([10, 10])),
                "2": Transformation(0.8, np.array([1, 20])),
                "3": Transformation(0.8, np.array([10, 11])),
            },
            2,
        ),
        (
            [
                {
                    "0": Transformation(0.2, np.array([5, 0])),
                    "1": Transformation(0.3, np.array([10, 10])),
                    "2": Transformation(0.8, np.array([30, 50])),
                },
                {
                    "2": Transformation(0.8, np.array([1, 20])),
                    "3": Transformation(0.8, np.array([10, 11])),
                },
            ],
            {
                "0": Transformation(0.2, np.array([5, 0])),
                "1": Transformation(0.3, np.array([10, 10])),
                "2": Transformation(0.8, np.array([1, 20])),
                "3": Transformation(0.8, np.array([10, 11])),
            },
            2,
        ),
    ],
)
def test_number_of_correct_components(
    pred_transformations: list[PieceTransformations],
    ground_truth: PieceTransformations,
    expected_components: int,
) -> None:
    assert (
        len(
            get_correctly_predicted_clusters(
                pred_transformations, ground_truth, 0.17, 5
            )
        )
        == expected_components
    )
