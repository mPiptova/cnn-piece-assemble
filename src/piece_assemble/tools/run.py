""" Script for running the clustering algorithm.

Example
--------
python src/piece_assemble/tools/run.py /path/to/config

"""


import argparse
import os
from multiprocessing import Pool

from image import load_bin_img, load_img
from piece_assemble.cluster import ClusterScorer
from piece_assemble.clustering import Clustering
from piece_assemble.config import load_config
from piece_assemble.descriptor import MultiOsculatingCircleDescriptor
from piece_assemble.piece import Piece
from piece_assemble.types import BinImg, NpImage


def load_images(
    img_dir: str, scale: float = 1
) -> tuple[list[str], list[NpImage], list[BinImg]]:
    """Load all piece images from the given directory.

    Parameters
    ----------
    img_dir
        Path to the directory containing piece images.
        Two images are required for every image, one PNG binary mask with "_mask" suffix
        and one JPG image with piece itself, both need to have the same size.
    scale
        All returned images will be rescaled by this factor.

    Returns
    -------
    img_ids
        A list of image names.
    imgs
        A list of 3-channel images of pieces
    masks
        A list of binary piece masks.
    """
    img_ids = [
        name
        for name in os.listdir(img_dir)
        if not name.endswith("_mask.png")
        and name not in ["pieces.json", "original.png"]
    ]

    imgs = [load_img(os.path.join(img_dir, name), scale) for name in img_ids]
    masks = [
        load_bin_img(os.path.join(img_dir, f"{name.split('.')[0]}_mask.png"), scale)
        for name in img_ids
    ]
    return img_ids, imgs, masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piece assemble.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    img_ids, imgs, masks = load_images(config["img_path"], config["piece"]["scale"])

    descriptor_extractor = MultiOsculatingCircleDescriptor(**config["descriptor"])

    with Pool(config["clustering"]["n_processes"]) as p:
        pieces = p.starmap(
            Piece,
            zip(
                img_ids,
                imgs,
                masks,
                [descriptor_extractor] * len(img_ids),
                [config["piece"]["sigma"]] * len(img_ids),
                [config["piece"]["polygon_approximation_tolerance"]] * len(img_ids),
            ),
        )

    cluster_scorer = ClusterScorer(**config["cluster_scorer"])
    clustering = Clustering(pieces, descriptor_extractor, cluster_scorer)

    clustering.set_logging(**config["logging"])
    clustering(
        **config["clustering"],
        cluster_config=config["cluster"],
        trusted_cluster_config=config["trusted_cluster"],
    )
