""" Script for running the clustering algorithm.

Example
--------
python src/piece_assemble/tools/run.py /path/to/config

"""


import argparse

from piece_assemble.cluster import EmbeddingClusterScorer
from piece_assemble.clustering import Clustering
from piece_assemble.config import load_config
from piece_assemble.load import load_images
from piece_assemble.models.predict import load_predictor
from piece_assemble.piece import Piece

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piece assemble.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    img_ids, imgs, masks = load_images(config["img_path"], config["piece"]["scale"])

    pieces = [
        Piece.from_image(
            img_name, img, mask, config["piece"]["polygon_approximation_tolerance"]
        )
        for img_name, img, mask in zip(img_ids, imgs, masks)
    ]
    pieces_dict = {piece.name: piece for piece in pieces}

    predictor = load_predictor(
        config["model"]["id"],
        config["model"]["directory"],
        config["model"]["activation_threshold"],
    )

    cluster_scorer = EmbeddingClusterScorer(
        predictor.predict_embeddings(pieces_dict), pieces_dict
    )
    clustering = Clustering(pieces, cluster_scorer)

    clustering.set_logging(**config["logging"])
    clustering(
        **config["clustering"],
        cluster_config=config["cluster"],
        trusted_cluster_config=config["trusted_cluster"],
        predictor=predictor,
    )
