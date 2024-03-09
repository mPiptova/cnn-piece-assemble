"""
This script performs the preprocessing of input images of pieces.

Usage
-----
python src/piece_assemble/tools/preprocess.py \
    [-h] input_path [input_path ...] output_dir
"""

import argparse
from os import path

import numpy as np
from PIL import Image
from PIL.Image import Image as PilImage
from tqdm import tqdm

from piece_assemble.image import np_to_pil
from piece_assemble.preprocessing import NegativePieceExtractor, PieceExtractorBase


def store_piece_image(img: PilImage, output_dir: str, basename: str) -> None:
    """Save the piece image to the given directory.

    The image will be stored as basename.jpg.

    Parameters
    ----------
    img
        Input PIL image.
    output_dir
        The output directory.
    basename
        The name of the image (without the extension).
    """
    dst_path = path.join(output_dir, f"{basename}.jpg")
    img.save(dst_path)


def store_piece_mask(mask: np.ndarray, output_dir: str, basename: str) -> None:
    """Save the piece binary mask to the given directory.

    The image will be stored as basename_mask.png.

    Parameters
    ----------
    mask
        Binary 2d array representing the piece mask.
        Value 1 indicates that the given pixel belongs to the
        segmented piece, 0 indicates background pixel.
    output_dir
        The output directory.
    basename
        The name of the image (without the extension).
    """
    img = np_to_pil(mask)
    dst_path = path.join(output_dir, f"{basename}_mask.png")
    img.save(dst_path)


def process_image(
    img_path: str, output_dir: str, piece_extractor: PieceExtractorBase
) -> None:
    """Preprocess one piece image and store it to the given location.

    Parameters
    ----------
    img_path
        The path to the input image.
    output_dir
        The path where preprocessed image will be stored.
    piece_extractor
        The piece extractor which should be used to preprocess the image.
    """
    basename = path.basename(img_path).split(".")[0]
    img = Image.open(img_path)

    piece_img, piece_mask = piece_extractor(img)

    store_piece_image(piece_img, output_dir, basename)
    store_piece_mask(piece_mask, output_dir, basename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument("inputs", nargs="+")
    parser.add_argument("output_dir")
    parser.add_argument("--preserve-holes", action="store_true")
    parser.add_argument("--background-var", type=float, default=1.5)
    parser.add_argument("--max-size", type=int, default=3000)

    args = parser.parse_args()
    piece_extractor = NegativePieceExtractor(
        args.background_var,
        fill_holes=not args.preserve_holes,
        max_image_size=args.max_size,
    )

    for img_path in tqdm(args.inputs):
        process_image(img_path, args.output_dir, piece_extractor)
