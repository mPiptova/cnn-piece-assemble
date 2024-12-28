import numpy as np
from perlin_noise import PerlinNoise
from skimage.measure import label
from skimage.morphology import binary_erosion, disk

from piece_assemble.piece import Piece, TransformedPiece
from puzzle_generator.plane_division import crop_piece_img


def generate_noise_image(shape: tuple[int, int], octaves: int = 30) -> np.ndarray:
    """Generate image containing Perlin noise.

    Parameters
    ----------
    shape
        The shape of the image to generate.
    octaves
        The number of octaves to use for the Perlin noise.

    Returns
    -------
    A numpy array of shape (shape[0], shape[1]) containing the generated noise image.
    """
    noise = PerlinNoise(octaves, seed=1)
    noise_array = [
        [noise([i / shape[0], j / shape[1]]) for j in range(shape[1])]
        for i in range(shape[0])
    ]
    noise_img = np.array(noise_array)

    noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())
    return noise_img


def _get_erosion_prob(mask: np.ndarray, width: int = 4) -> np.ndarray:
    dil_prob_mask = mask.astype(float)
    dil_mask = mask.astype(bool)
    footprint = disk(1)
    step = 1 / width
    for i in range(width):
        dil_mask = binary_erosion(dil_mask, footprint=footprint)
        dil_prob_mask[dil_mask] = (width - i - 1) * step

    return dil_prob_mask


def _get_random_noise_crop(img_noise: np.ndarray, shape: tuple) -> np.ndarray:
    i = np.random.randint(0, img_noise.shape[0] - shape[0])
    j = np.random.randint(0, img_noise.shape[1] - shape[1])
    return img_noise[i : i + shape[0], j : j + shape[1]]


def _get_eroded_mask(
    mask: np.ndarray, img_noise: np.ndarray, width: int = 5
) -> np.ndarray:
    erosion_prob_mask = _get_erosion_prob(mask, width=width)
    img_noise_mask = _get_random_noise_crop(img_noise, mask.shape)
    erosion_prob = erosion_prob_mask * img_noise_mask

    new_mask = (erosion_prob < 0.5) & mask

    return new_mask.astype(bool)


def apply_random_erosion(
    mask: np.ndarray,
    img: np.ndarray,
    noise_img: np.ndarray | None = None,
    strength: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random erosion to piece image, simulating imperfectly matching puzzle.

    Parameters
    ----------
    mask
        The mask of the puzzle piece.
    img
        The image of the puzzle piece.
    noise_img
        The noise image to use for the erosion. If None, a new noise image will
        be generated.
    strength
        The strength of the erosion.

    Returns
    -------
    A tuple containing the eroded mask and the eroded image.

    """
    if noise_img is None:
        noise_img = generate_noise_image(mask.shape, octaves=15)

    eroded_mask = _get_eroded_mask(mask, noise_img, strength)

    labels = label(eroded_mask)
    if labels.max() > 1:
        vals, counts = np.unique(labels, return_counts=True)
        counts = counts[vals != 0]
        vals = vals[vals != 0]
        eroded_mask = labels == vals[np.argmax(counts)]

    eroded_img = img.copy()
    eroded_img[~eroded_mask] = 1
    return eroded_mask, eroded_img


def apply_random_erosion_to_pieces(
    pieces: list[TransformedPiece],
    strength: int = 6,
) -> list[Piece]:
    """Apply random erosion to the mask, simulating imperfectly matching puzzle.

    Parameters
    ----------
    pieces
        The pieces to apply the erosion to.
    strength
        The strength of the erosion.

    Returns
    -------
    A tuple containing the eroded masks, images and transformations.
    """
    noise_img = generate_noise_image((2000, 2000))
    new_pieces = []

    for piece in pieces:
        eroded_mask, eroded_img = apply_random_erosion(
            piece.mask, piece.img, noise_img, strength
        )
        eroded_img, eroded_mask, transformation_i = crop_piece_img(
            eroded_img, eroded_mask, piece.transformation.inverse()
        )

        new_piece = Piece(
            piece.name,
            eroded_img,
            piece.img_avg,
            eroded_mask,
            piece.contour,
            piece.feature_extractor,
            piece.features,
            piece.holes,
            piece.hole_features,
            piece.polygon,
        )

        new_pieces.append(TransformedPiece(new_piece, transformation_i))

    return new_pieces
