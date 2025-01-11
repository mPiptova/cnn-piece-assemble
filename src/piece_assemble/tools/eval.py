import argparse
import os

from tqdm import tqdm

from piece_assemble.load import load_puzzle
from piece_assemble.models import load_model
from piece_assemble.models.eval import eval_puzzles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("puzzles_path", type=str)
    parser.add_argument("activation_threshold", type=float)
    parser.add_argument("models_path", type=str)
    parser.add_argument("models", type=str, nargs="+")
    args = parser.parse_args()

    puzzles = {}

    with open(args.puzzles_path, "r") as f:
        puzzle_dirs = f.readlines()
        puzzle_dirs = [puzzle_dir.strip() for puzzle_dir in puzzle_dirs]

    header = (
        "model, dataset, threshold, puzzles, n_puzzles, macro_precision, "
        + "macro_recall, macro_f1, macro_accuracy, fa, lcc, tp, tn, fp, fn"
    )
    output = [header]
    for model_id in args.models:
        model = load_model(model_id, args.models_path)

        if not puzzles.get(model.background_val, False):
            puzzles[model.background_val] = [
                load_puzzle(
                    os.path.join(args.dataset_path, puzzle_dir), model.background_val
                )
                for puzzle_dir in tqdm(puzzle_dirs, desc="Loading puzzles")
            ]

        metrics = eval_puzzles(
            model, puzzles[model.background_val], args.activation_threshold
        )
        output.append(
            f"{model_id}, {args.dataset_path}, {args.activation_threshold}, "
            + f"{args.puzzles_path}, {len(puzzle_dirs)}, {metrics['precision']}, "
            + f"{metrics['recall']}, {metrics['f1']}, {metrics['accuracy']}, "
            + f"{metrics['fa']}, {metrics['lcc']}, {metrics['tp']}, {metrics['tn']}, "
            + f"{metrics['fp']}, {metrics['fn']}"
        )

    for line in output:
        print(line)
