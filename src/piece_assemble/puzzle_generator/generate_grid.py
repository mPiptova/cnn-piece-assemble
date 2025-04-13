import argparse
import os

import numpy as np
from PIL import Image

from piece_assemble.image import pil_to_np
from piece_assemble.puzzle_generator.generate import resize_image, store_puzzle
from piece_assemble.puzzle_generator.plane_division import apply_division_to_image


def get_grid_division(rows: int, columns: int, shape: tuple[int, int]) -> np.ndarray:
    division_img = np.zeros(shape, dtype=np.uint8)
    step_rows = shape[0] // rows
    step_columns = shape[1] // columns
    counter = 1
    for i in range(rows):
        for j in range(columns):
            division_img[
                i * step_rows : (i + 1) * step_rows,
                j * step_columns : (j + 1) * step_columns,
            ] = counter
            counter += 1

    return division_img


def generate_puzzle(rows: int, columns: int, img: np.ndarray, output_dir: str) -> None:
    # crop image to multiple of rows and columns
    img = img[
        : img.shape[0] - img.shape[0] % rows, : img.shape[1] - img.shape[1] % columns
    ]
    rng = np.random.default_rng()
    division = get_grid_division(rows, columns, img.shape[:2])
    pieces = apply_division_to_image(img, division, rng)

    piece_dict = {piece.name: piece for piece in pieces}

    store_puzzle(piece_dict, output_dir, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Puzzle generator.")

    parser.add_argument(
        "rows",
        type=int,
        help="Number of pieces in one row",
    )

    parser.add_argument(
        "columns",
        type=int,
        help="Number of pieces in one column",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum size of the larger side of the image",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory",
    )

    parser.add_argument(
        "img_paths",
        type=str,
        nargs="+",
        help="Image paths",
    )

    args = parser.parse_args()

    for i, img_path in enumerate(args.img_paths):
        print(f"Processing image {i + 1}/{len(args.img_paths)}: {img_path}")
        img_pil = Image.open(img_path)
        img_pil = resize_image(args.max_size, img_pil)

        output_dir = os.path.join(
            args.output_dir,
            f"{args.rows * args.columns}_{os.path.basename(img_path)[:-4]}",
        )
        if os.path.exists(output_dir):
            print(f"Skipping {output_dir}, already exists")
            continue

        os.makedirs(output_dir)

        img = pil_to_np(img_pil)
        generate_puzzle(
            args.rows,
            args.columns,
            img,
            output_dir,
        )
