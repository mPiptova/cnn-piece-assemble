""" Script for running the clustering algorithm.

Example
--------
python src/piece_assemble/tools/run.py /path/to/config

"""


import argparse
from multiprocessing import Pool

from piece_assemble.cluster import ClusterScorer
from piece_assemble.clustering import Clustering
from piece_assemble.config import load_config
from piece_assemble.feature_extraction.segment import (
    MultiOsculatingCircleFeatureExtractor,
)
from piece_assemble.load import load_images
from piece_assemble.piece import Piece

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piece assemble.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    img_ids, imgs, masks = load_images(config["img_path"], config["piece"]["scale"])

    feature_extractor = MultiOsculatingCircleFeatureExtractor(**config["descriptor"])

    with Pool(config["clustering"]["n_processes"]) as p:
        pieces = p.starmap(
            Piece,
            zip(
                img_ids,
                imgs,
                masks,
                [feature_extractor] * len(img_ids),
                [config["piece"]["sigma"]] * len(img_ids),
                [config["piece"]["polygon_approximation_tolerance"]] * len(img_ids),
            ),
        )

    cluster_scorer = ClusterScorer(**config["cluster_scorer"])
    clustering = Clustering(pieces, feature_extractor, cluster_scorer)

    clustering.set_logging(**config["logging"])
    clustering(
        **config["clustering"],
        cluster_config=config["cluster"],
        trusted_cluster_config=config["trusted_cluster"],
    )
