import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from piece_assemble.dataset import BatchCollator, PairsDataset
from piece_assemble.dataset.create import load_puzzle
from piece_assemble.models import PairNetwork
from piece_assemble.models.train import MaskedBCEWithLogitsLoss, train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the directory where checkpoints are stored",
    )
    parser.add_argument(
        "tensorboard_path",
        type=str,
        help="Path to the directory where checkpoints are stored",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"UNet_{timestamp}"

    config["id"] = f"UNet_{timestamp}"

    with open(os.path.join(args.checkpoint_path, f"{model_id}_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = PairNetwork(**config["model"])
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["learning_rate"]
    )
    pos_weight = (
        torch.tensor(config["train"]["pos_weight"])
        if config["train"]["pos_weight"] is not None
        else None
    )
    loss_fn = MaskedBCEWithLogitsLoss(
        pos_weight=pos_weight, subsample_neg_ratio=config["train"]["loss_neg_ratio"]
    )
    loss_fn = loss_fn.to(device)

    print(f"Model parameters: {sum([par.numel() for par in model.parameters()])}")

    val_dataset = PairsDataset(
        f"{config['train']['dataset']}/val",
        model.padding,
        batch_size=config["train"]["batch_size"],
        negative_ratio=config["train"]["negative_ratio"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=BatchCollator(model.padding, 2 ** config["model"]["depth"]),
    )

    dataset = PairsDataset(
        f"{config['train']['dataset']}/train",
        model.padding,
        batch_size=config["train"]["batch_size"],
        negative_ratio=config["train"]["negative_ratio"],
    )
    training_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=BatchCollator(model.padding, 2 ** config["model"]["depth"]),
    )

    test_examples = ["100_1952.5.34", "50_1942.8.38", "50_1943.3.4716.q"]
    puzzles = [
        load_puzzle(
            f"/root/rir/project/data/val/{example}/", config["model"]["background_val"]
        )
        for example in test_examples
    ]

    writer = SummaryWriter(os.path.join(args.tensorboard_path, model_id))
    print(f"Training model {model_id}")

    train_model(
        model,
        model_id,
        loss_fn,
        optimizer,
        training_loader,
        val_loader,
        config["train"]["epochs"],
        writer,
        0,
        puzzles=puzzles,
        save_path=args.checkpoint_path,
    )
