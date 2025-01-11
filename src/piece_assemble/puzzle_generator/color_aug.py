from __future__ import annotations

from typing import TYPE_CHECKING

import albumentations as albu
import numpy as np

if TYPE_CHECKING:
    from albumentations.core.transforms_interface import BasicTransform


def augment_image(
    img: np.ndarray, mask: np.ndarray, augmentation: BasicTransform
) -> np.ndarray:
    img = (img * 255).astype(np.uint8)
    augmented = augmentation(image=img)["image"]
    augmented = (augmented / 255).astype(np.float32)
    augmented[~mask.astype(bool)] = 1

    return augmented


def color_augmentation(
    img: np.ndarray,
    mask: np.ndarray,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.05,
    p: float = 0.5,
) -> np.ndarray:
    aug = albu.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=p
    )

    return augment_image(img, mask, aug)
