import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class MaskedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        subsample_neg_ratio: float | None = None,
    ) -> None:
        super().__init__(pos_weight=pos_weight)
        self.subsample_neg_ratio = subsample_neg_ratio

    def _get_subsample_mask(self, target: torch.Tensor) -> torch.Tensor:
        mask = target == 1
        neg_count = int(mask.sum().item() * self.subsample_neg_ratio)
        if neg_count == 0:
            return torch.ones_like(mask, dtype=torch.bool)
        perm = torch.randperm(target.size(0))
        idx = perm[:neg_count]

        mask[idx] = True
        return mask

    def mask(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target != -1
        return input[mask], target[mask]

    def subsample(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.subsample_neg_ratio is not None:
            mask = self._get_subsample_mask(target)
            input = input[mask]
            target = target[mask]

        return input, target

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input, target = self.mask(input, target)
        input, target = self.subsample(input, target)
        return super().__call__(input, target)


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    tb_writer: SummaryWriter,
    epoch_index: int,
    log_interval: int = 50,
):
    """
    Trains the model for one epoch.
    """
    running_loss = 0.0

    with tqdm(total=len(train_loader)) as pbar:
        for i, data in enumerate(train_loader):
            pbar.update(1)
            (input1, input2), labels = data

            device = next(model.parameters()).device

            input1 = input1.to(device)
            input2 = input2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model((input1, input2))

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            pbar.set_description(
                f"Epoch {epoch_index}: training Loss: {loss.item():0.4f}"
            )

            if i % log_interval == log_interval - 1:
                tb_x = epoch_index * len(train_loader) + i + 1
                tb_writer.add_scalar("Loss/train", running_loss / log_interval, tb_x)
                running_loss = 0.0
                tb_writer.flush()

    return running_loss / (i % log_interval + 1)


def evaluate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
):
    model.eval()

    running_loss = 0
    running_precision = 0
    running_recall = 0
    running_f1 = 0
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(val_loader), desc="Evaluating", total=len(val_loader)
        ):
            (input1, input2), labels = data

            device = next(model.parameters()).device
            input1 = input1.to(device)
            input2 = input2.to(device)
            labels = labels.to(device)

            outputs = model((input1, input2))

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            masked_output, masked_labels = loss_fn.mask(outputs, labels)

            tp = torch.sum((masked_output > 0.5) & (masked_labels == 1))
            fp = torch.sum((masked_output > 0.5) & (masked_labels == 0))
            fn = torch.sum((masked_output <= 0.5) & (masked_labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            running_precision += precision
            running_recall += recall
            running_f1 += f1

    avg_vloss = running_loss / (i + 1)
    avg_precision = running_precision / (i + 1)
    avg_recall = running_recall / (i + 1)
    avg_f1 = running_f1 / (i + 1)

    return {
        "loss": avg_vloss,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }


def train_model(
    model: torch.nn.Module,
    model_id: str,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    tb_writer: torch.utils.tensorboard.SummaryWriter,
    start_epoch: int = 0,
):
    epoch_number = start_epoch
    for epoch_number in range(epoch_number, epoch_number + epochs):
        model.train(True)
        avg_loss = train_one_epoch(
            model, loss_fn, optimizer, train_loader, tb_writer, epoch_number
        )

        eval_results = evaluate(model, val_loader, loss_fn)
        tb_writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": eval_results["loss"]},
            epoch_number,
        )
        tb_writer.add_scalar("Validation Recall", eval_results["recall"], epoch_number)
        tb_writer.add_scalar(
            "Validation Precision", eval_results["precision"], epoch_number
        )
        tb_writer.add_scalar("Validation F1", eval_results["f1"], epoch_number)
        tb_writer.flush()

        # # # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        model_path = f"{model_id}_{epoch_number}"
        torch.save(model.state_dict(), model_path)

        tb_writer.flush()
