import numpy as np
from PIL import Image

from piece_assemble.types import NpImage, PilImage


def pil_to_np(img: PilImage) -> NpImage:
    """Convert PIL image to numpy array.

    Parameters
    ----------
    img
        PIL image.

    Returns
    -------
    Numpy array.
    """
    return np.array(img) / 255


def np_to_pil(img: NpImage) -> PilImage:
    """Convert numpy array to PIL image.

    Parameters
    ----------
    img
        Numpy array representing an image.

    Returns
    -------
    PIL image.
    """
    return Image.fromarray((img * 255).astype(np.uint8))
