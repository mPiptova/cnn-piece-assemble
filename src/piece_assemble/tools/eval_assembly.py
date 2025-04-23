"""
Script for evaluating the assembly algorithm.

usage: eval_assembly.py [-h] config puzzles_path dataset_path

positional arguments:
  config        Configuration file
  puzzles_path  Path to the file containing the list of puzzles which should be used for evaluation
  dataset_path  Path to the directory where the puzzles are stored
"""

import argparse
import os
from copy import deepcopy

from piece_assemble.config import load_config
from piece_assemble.evaluation import cluster_to_transformations, evaluate
from piece_assemble.load import load_puzzle
from piece_assemble.tools.run_assembly import run_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate piece assemble.")
    parser.add_argument("config", type=str, help="Configuration file")
    parser.add_argument("puzzles_path", type=str, help="Path to the file containing the list of puzzles which should be used for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to the directory where the puzzles are stored")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    with open(args.puzzles_path, "r") as f:
        puzzle_dirs = f.readlines()
        puzzle_dirs = [
            os.path.join(args.dataset_path, puzzle_dir.strip())
            for puzzle_dir in puzzle_dirs
        ]

    eval_results = {}

    for puzzle_dir in puzzle_dirs:
        pieces, neighbors = load_puzzle(puzzle_dir)
        gt_transformations = {
            piece.name: piece.transformation for piece in pieces.values()
        }

        copy_config = deepcopy(config)
        copy_config["img_path"] = puzzle_dir
        if config["logging"].get("output_images_path", False):
            copy_config["logging"]["output_images_path"] = os.path.join(
                config["logging"]["output_images_path"], puzzle_dir.split("/")[-1]
            )

        pred_clusters = [
            cluster_to_transformations(cluster)
            for cluster in run_prediction(copy_config)
        ]
        metrics = evaluate(
            pred_clusters,
            gt_transformations,
            neighbors,
            config["cluster"]["rotation_tol"],
            config["cluster"]["translation_tol"],
        )
        print(puzzle_dir, metrics)

        eval_results[puzzle_dir] = metrics

    aggregated_metrics = {}
    for metric in metrics.keys():
        aggregated_metrics[metric] = sum(
            [metrics[metric] for metrics in eval_results.values()]
        ) / len(eval_results)

    print(",".join(["puzzle"] + list(metrics.keys())))
    for puzzle_dir, metrics in eval_results.items():
        print(",".join([puzzle_dir] + [str(metrics[key]) for key in metrics.keys()]))

    print(
        ",".join(["aggregated"] + [str(value) for value in aggregated_metrics.values()])
    )
