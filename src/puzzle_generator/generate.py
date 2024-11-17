"""Script for generating puzzles form pictures.

Usage
-----
```
python generate.py NUM_PIECES \
    [--num-divisions NUM_DIVISIONS] \
    [--num-samples NUM_SAMPLES] \
    [--perturbation-strength PERTURBATION_STRENGTH] \
    [--max-size MAX_SIZE] \
    OUTPUT_DIR \
    IMG1 [IMG2 ...]
```

Following command will generate puzzle with 50 pieces with default
parameters from images path/to/image1.jpg and path/to/image2.jpg and
store it in output/dir

```
python generate.py 50 output/dir path/to/image1.jpg path/to/image2.jpg
```

Following command will generate puzzle with custom parameters

```
python generate.py 50 \
    --num-divisions 10 \
    --num-samples 10 \
    --perturbation-strength 5 \
    output/dir path/to/image1.jpg path/to/image2.jpg
```
"""

# !/usr/bin/env python
import argparse
import json
import math
import os

from PIL import Image

from image import np_to_pil, pil_to_np
from piece_assemble.cluster import Cluster, DummyClusterScorer
from piece_assemble.neighbors import BorderLengthNeighborClassifier
from piece_assemble.types import NpImage
from puzzle_generator.plane_division import (
    apply_division_to_image,
    get_random_division,
    reduce_number_of_pieces,
)


def generate_puzzle(
    img: NpImage,
    num_pieces: int,
    num_divisions: int,
    num_samples: int,
    perturbation_strength: int,
    output_dir: str,
) -> None:
    division = get_random_division(
        img.shape[0], img.shape[1], num_divisions, num_samples, perturbation_strength
    )
    division = reduce_number_of_pieces(division, num_pieces, 1000)
    pieces = apply_division_to_image(img, division)

    piece_dict = {piece.piece.name: piece for piece in pieces}

    # Write original
    np_to_pil(img).save(os.path.join(output_dir, "original.png"))

    # Write images
    for name, piece in piece_dict.items():
        np_to_pil(piece.piece.img).save(os.path.join(output_dir, f"{name}.png"))
        np_to_pil(piece.piece.mask).save(os.path.join(output_dir, f"{name}_mask.png"))

    # Write cluster
    cluster = Cluster(
        piece_dict,
        DummyClusterScorer(),
        0,
        0,
        0,
        0,
        BorderLengthNeighborClassifier(30, 5),
        None,
    )
    with open(os.path.join(output_dir, "pieces.json"), "w") as f:
        json.dump(cluster.to_dict(), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Puzzle generator.")

    parser.add_argument(
        "num_pieces",
        type=int,
        help="Number of pieces to assemble",
    )

    parser.add_argument(
        "--num-divisions",
        type=int,
        default=None,
        help="Number of divisions of puzzle",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples",
    )

    parser.add_argument(
        "--perturbation-strength",
        type=int,
        default=None,
        help="Perturbation strength",
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

    # Set defaults
    if args.num_divisions is None:
        args.num_divisions = int(args.num_pieces**0.8)

    if args.num_samples is None:
        args.num_samples = int(math.sqrt(args.num_pieces)) * 2

    for i, img_path in enumerate(args.img_paths):
        print(f"Processing image {i + 1}/{len(args.img_paths)}: {img_path}")
        img_pil = Image.open(img_path)

        if args.max_size is not None and (
            img_pil.width > args.max_size or img_pil.height > args.max_size
        ):
            new_shape = (
                (int(img_pil.width * args.max_size / img_pil.height), args.max_size)
                if img_pil.height > img_pil.width
                else (
                    args.max_size,
                    int(img_pil.height * args.max_size / img_pil.width),
                )
            )
            img_pil = img_pil.resize(new_shape)

        output_dir = os.path.join(
            args.output_dir, f"{args.num_pieces}_{os.path.basename(img_path)[:-4]}"
        )
        if os.path.exists(output_dir):
            print(f"Skipping {output_dir}, already exists")
            continue

        os.makedirs(output_dir)

        img = pil_to_np(img_pil)
        generate_puzzle(
            img,
            args.num_pieces,
            args.num_divisions,
            args.num_samples,
            args.perturbation_strength,
            output_dir,
        )
