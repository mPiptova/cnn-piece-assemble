import numpy as np
from PIL import Image
from skimage.transform import rescale

from piece_assemble.types import BinImg, NpImage, PilImage


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


def load_bin_img(img_path: str, scale: float = 1, padding: int = 1) -> BinImg:
    """Load image as a binary array.

    Parameters
    ----------
    img_path
    scale
        If not 1, the image will be rescaled by given factor.
    padding
        Padding size. Padding value is 0 (background).

    Returns
    -------
    bin_img
    """
    img = Image.open(img_path)
    img_np = rescale(pil_to_np(img).astype(bool), scale).astype(np.uint8)
    return np.pad(img_np, padding, mode="constant", constant_values=0)


def load_img(img_path: str, scale: float = 1, padding: int = 1) -> NpImage:
    """Load image as numpy image.

    Parameters
    ----------
    img_path
    scale
        If not 1, the image will be rescaled by given factor.
    padding
        Padding size. Padding value is 1 (white).

    Returns
    -------
    bin_img
    """
    img = Image.open(img_path)
    img_np = rescale(pil_to_np(img), scale, channel_axis=2)
    return np.pad(
        img_np,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="constant",
        constant_values=1,
    )
