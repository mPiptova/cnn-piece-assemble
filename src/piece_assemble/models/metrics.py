"""This module contains definitions of various metrics used for evaluation."""

from abc import abstractmethod

import numpy as np

from piece_assemble.geometry import Transformation


class Metric:
    """Abstract metric"""
    @abstractmethod
    def __call__(
        self, pred: Transformation | None, gold: Transformation | None
    ) -> float | None:
        ...


class RotationAngleError(Metric):
    """Metric measuring the rotation error of transformation"""
    def __call__(
        self, pred: Transformation | None, gold: Transformation | None
    ) -> float | None:
        """Compute rotation error between two transformations"""
        if pred is None or gold is None:
            return None
        diff = pred.rotation_angle - gold.rotation_angle
        err: float = abs((diff + np.pi) % (2 * np.pi) - np.pi)
        return err


class TranslationError(Metric):
    """Metric measuring the translation error of transformation"""
    def __call__(
        self, pred: Transformation | None, gold: Transformation | None
    ) -> float | None:
        """Compute translation error between two transformations"""
        if pred is None or gold is None:
            return None
        err: float = np.linalg.norm(pred.translation - gold.translation)
        return err


class TransformationError(Metric):
    """Returns the error of predicted transformation.

    Weighted average of angle and translation errors is used."""

    def __init__(
        self,
        angle_tol: float = 0.17,
        translation_tol: float = 30,
        translation_weight: float = 1,
    ):
        self.angle_tol = angle_tol
        self.translation_tol = translation_tol
        self.correctness_classifier = CorrectnessClassifier(angle_tol, translation_tol)
        self.translation_weight = translation_weight

    def __call__(
        self, pred: Transformation | None, gold: Transformation | None
    ) -> float | None:
        """Compute the error of transformation"""
        if pred is None or gold is None:
            return None
        if not self.correctness_classifier(pred, gold):
            return None

        angle_error = RotationAngleError()(pred, gold)
        translation_error = TranslationError()(pred, gold)

        normalized_angle_error = angle_error / self.angle_tol
        normalized_translation_error = translation_error / self.translation_tol

        combined_error = (
            normalized_angle_error
            + normalized_translation_error * self.translation_weight
        )
        max_error = 1 + self.translation_weight

        return combined_error / max_error


class CorrectnessClassifier:
    """Returns True if the predicted transformation is correct, False otherwise"""

    def __init__(self, angle_tol: float = 0.17, translation_tol: float = 30):
        self.angle_tol = angle_tol
        self.translation_tol = translation_tol

    def __call__(
        self, pred: Transformation | None, gold: Transformation | None
    ) -> bool:
        if pred is None and gold is None:
            return True
        if pred is None or gold is None:
            return False
        result: bool = pred.is_close(gold, self.angle_tol, self.translation_tol)
        return result
