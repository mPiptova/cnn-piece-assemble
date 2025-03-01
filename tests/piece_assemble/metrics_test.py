import numpy as np
import pytest

from piece_assemble.geometry import Transformation
from piece_assemble.models.metrics import RotationAngleError


@pytest.mark.parametrize(
    "pred_transformation, true_transformation, error ",
    [
        (Transformation.identity(), Transformation.identity(), 0),
        (Transformation.identity(), Transformation(np.pi, np.array([0, 0])), np.pi),
        (
            Transformation(0.3, np.array([0, 0])),
            Transformation(0.6, np.array([0, 0])),
            0.3,
        ),
        (
            Transformation(0.6, np.array([0, 0])),
            Transformation(0.3, np.array([0, 0])),
            0.3,
        ),
        (
            Transformation(2 * np.pi - 0.1, np.array([0, 0])),
            Transformation(0.1, np.array([0, 0])),
            0.2,
        ),
        (
            Transformation(0.1, np.array([0, 0])),
            Transformation(2 * np.pi - 0.1, np.array([0, 0])),
            0.2,
        ),
        (
            Transformation(-np.pi, np.array([0, 0])),
            Transformation(np.pi, np.array([0, 0])),
            0,
        ),
        (
            Transformation(0, np.array([0, 0])),
            Transformation(2 * np.pi, np.array([0, 0])),
            0,
        ),
    ],
)
def test_rotation_error(
    pred_transformation: Transformation,
    true_transformation: Transformation,
    error: float,
) -> None:
    assert np.isclose(
        RotationAngleError()(pred_transformation, true_transformation),
        error,
        rtol=1e-6,
    )
