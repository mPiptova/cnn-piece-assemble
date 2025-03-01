import argparse
import os

from tqdm import tqdm

from piece_assemble.load import load_puzzle
from piece_assemble.models.eval import eval_puzzles
from piece_assemble.models.metrics import (
    RotationAngleError,
    TransformationError,
    TranslationError,
)
from piece_assemble.models.predict import load_predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("puzzles_path", type=str)
    parser.add_argument("activation_threshold", type=float)
    parser.add_argument("models_path", type=str)
    parser.add_argument("models", type=str, nargs="+")
    args = parser.parse_args()

    with open(args.puzzles_path, "r") as f:
        puzzle_dirs = f.readlines()
        puzzle_dirs = [puzzle_dir.strip() for puzzle_dir in puzzle_dirs]

    puzzles = [
        load_puzzle(os.path.join(args.dataset_path, puzzle_dir))
        for puzzle_dir in tqdm(puzzle_dirs, desc="Loading puzzles")
    ]

    angle_error = RotationAngleError()
    translation_error = TranslationError()
    transformation_error = TransformationError()

    additional_metrics = {
        "angle_error": angle_error,
        "translation_error": translation_error,
        "transformation_error": transformation_error,
    }

    header = (
        "model, dataset, threshold, puzzles, n_puzzles, macro_precision, "
        + "macro_recall, macro_f1, macro_accuracy, fa, lcc, tp, tn, fp, fn, "
        + ", ".join(additional_metrics.keys())
    )
    output = [header]
    for predictor_id in args.models:
        predictor = load_predictor(
            predictor_id, args.models_path, args.activation_threshold
        )

        metrics = eval_puzzles(
            predictor, puzzles, additional_metrics=additional_metrics
        )
        output.append(
            f"{predictor_id}, {args.dataset_path}, {args.activation_threshold}, "
            + f"{args.puzzles_path}, {len(puzzle_dirs)}, {metrics['precision']}, "
            + f"{metrics['recall']}, {metrics['f1']}, {metrics['accuracy']}, "
            + f"{metrics['fa']}, {metrics['lcc']}, {metrics['tp']}, {metrics['tn']}, "
            + f"{metrics['fp']}, {metrics['fn']}, "
            + ", ".join([str(metrics[key]) for key in additional_metrics.keys()])
        )

    for line in output:
        print(line)
