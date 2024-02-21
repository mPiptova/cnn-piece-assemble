from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from PIL.Image import Image


def pil_to_np(img: Image) -> np.array:
    return np.array(img) / 255


def np_to_pil(img: np.ndarray) -> np.ndarray:
    return Image.fromarray((img * 255).astype(np.uint8))
