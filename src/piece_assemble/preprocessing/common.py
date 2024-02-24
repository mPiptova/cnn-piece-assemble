import numpy as np
from PIL import Image


def pil_to_np(img: Image) -> np.array:
    return np.array(img) / 255


def np_to_pil(img: np.ndarray) -> np.ndarray:
    return Image.fromarray((img * 255).astype(np.uint8))


def get_resize_shape(original_shape: tuple[int, int], max_size: str) -> tuple[int, int]:
    if max(original_shape) <= max_size:
        return original_shape

    if original_shape[0] > original_shape[1]:
        new_shape = (
            max_size,
            round(original_shape[1] * (max_size / original_shape[0])),
        )
    else:
        new_shape = (
            round(original_shape[0] * (max_size / original_shape[1])),
            max_size,
        )

    return new_shape
