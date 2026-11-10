"""Training loop for HAM10000 EfficientNet-B0 transfer learning."""

import argparse
import csv
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ml.src.dataloaders import get_dataloaders
from ml.src.metrics import compute_metrics
from ml.src.model import build_model
from ml.src.registry import MODELS_DIR, RESULTS_DIR, checkpoint_path

# Append-only log of every epoch across every run, so results from different
# training experiments (hyperparams, architecture tweaks, etc.) can be
# compared later in pandas/Excel instead of only living in terminal scrollback.
# The version column says which model (= which checkpoint file) a row belongs to.
METRICS_LOG_PATH = RESULTS_DIR / "training_log.csv"
METRICS_FIELDS = [
    "version",
    "seed",
    "timestamp",
    "stage",
    "epoch",
    "num_epochs",
    "lr",
    "batch_size",
    "train_loss",
    "train_bal_acc",
    "train_f1",
    "val_loss",
    "val_bal_acc",
    "val_f1",
    "val_acc",
    "val_mel_recall",
    "is_best",
]


def log_metrics_row(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new_file = not METRICS_LOG_PATH.exists()
    with open(METRICS_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Fixes every random number generator training touches (head init,
    WeightedRandomSampler order, augmentations, dropout), so a run with the
    same seed and settings can be repeated. GPU kernels can still add tiny
    non-determinism, so repeats are close but not always bit-identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# No class weights here: the WeightedRandomSampler in get_dataloaders() already
# rebalances what the model sees during training, so adding loss-level class
# weights too would double-correct for the nv/df/vasc imbalance.
criterion = nn.CrossEntropyLoss()


def run_epoch(model, dataloader, optimizer=None):
    """One pass over dataloader. Trains if optimizer is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in tqdm(dataloader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, compute_metrics(all_labels, all_preds)


def run_training(
    model,
    train_dl,
    val_dl,
    optimizer,
    num_epochs,
    stage,
    lr,
    batch_size,
    version,
    seed,
    best_val_bal_acc=0.0,
    scheduler=None,
    patience=None,
):
    """Shared epoch loop: trains, validates, checkpoints on val balanced
    accuracy improvement to this version's own checkpoint file, and logs every
    epoch to METRICS_LOG_PATH.

    best_val_bal_acc lets fine_tune() seed this with stage 1's score, so stage 2
    only overwrites the checkpoint if it actually beats the frozen-backbone model.

    scheduler (optional) is stepped once per epoch. patience (optional) is early
    stopping: training ends once val balanced accuracy hasn't improved for that
    many epochs in a row -- the model has stopped getting better, so further
    epochs would only cost time (and risk overfitting the train set).
    """
    epochs_without_improvement = 0
    for epoch in range(1, num_epochs + 1):
        # LR actually used this epoch (changes each epoch under a scheduler).
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_m = run_epoch(model, train_dl, optimizer)
        val_loss, val_m = run_epoch(model, val_dl)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_bal_acc={train_m['balanced_accuracy']:.4f} | "
            f"val_loss={val_loss:.4f} val_bal_acc={val_m['balanced_accuracy']:.4f} "
            f"val_f1={val_m['macro_f1']:.4f} val_mel_recall={val_m['mel_recall']:.4f}"
        )

        # Balanced accuracy (mean per-class recall) picks the checkpoint: it
        # rewards catching every class, mel included, rather than just nv.
        is_best = val_m["balanced_accuracy"] > best_val_bal_acc
        if is_best:
            best_val_bal_acc = val_m["balanced_accuracy"]
            torch.save(model.state_dict(), checkpoint_path(version))
            print(f"  -> new best val_bal_acc={best_val_bal_acc:.4f}, saved checkpoint")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_metrics_row(
            {
                "version": version,
                "seed": seed,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "epoch": epoch,
                "num_epochs": num_epochs,
                "lr": f"{lr:.2e}",
                "batch_size": batch_size,
                "train_loss": f"{train_loss:.4f}",
                "train_bal_acc": f"{train_m['balanced_accuracy']:.4f}",
                "train_f1": f"{train_m['macro_f1']:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_bal_acc": f"{val_m['balanced_accuracy']:.4f}",
                "val_f1": f"{val_m['macro_f1']:.4f}",
                "val_acc": f"{val_m['accuracy']:.4f}",
                "val_mel_recall": f"{val_m['mel_recall']:.4f}",
                "is_best": is_best,
            }
        )

        if scheduler is not None:
            scheduler.step()
        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break

    return best_val_bal_acc


def train(
    version: str,
    seed: int = 42,
    num_epochs: int = 15,
    freeze_backbone: bool = True,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    """Stage 1: train only the classifier head on top of frozen ImageNet features."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Every version gets its own file, and an existing one is never
    # overwritten -- a typo'd or reused version name would otherwise silently
    # destroy an earlier model.
    if checkpoint_path(version).exists():
        raise FileExistsError(
            f"{checkpoint_path(version)} already exists -- pick a new version "
            f"name (e.g. the next vN) instead of overwriting an earlier model."
        )

    set_seed(seed)
    train_dl, val_dl, _ = get_dataloaders(batch_size=batch_size)
    model = build_model(freeze_backbone=freeze_backbone).to(DEVICE)

    # Only pass trainable params to the optimizer; with freeze_backbone=True
    # that's just the classifier head.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    return run_training(
        model, train_dl, val_dl, optimizer, num_epochs, "train", lr, batch_size, version, seed
    )


def fine_tune(
    version: str,
    seed: int = 42,
    num_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    cosine: bool = True,
    patience: int | None = 6,
):
    """Stage 2: unfreeze the whole network and keep training from stage 1's
    best weights, at a lower LR than stage 1 (1e-3 -> 1e-4), so the pretrained
    filters get adapted to skin lesions rather than wrecked by large updates.

    Defaults are v2's settings. v1 used lr=1e-5 for a fixed 10 epochs with no
    scheduler and no early stopping -- val score was still rising at the end,
    i.e. it stopped too early and learned too slowly. Cosine annealing lowers
    the LR smoothly from lr towards 0 over num_epochs: big steps early, fine
    adjustments at the end. Early stopping (patience) ends training once the
    val score stops improving.
    """
    set_seed(seed)
    train_dl, val_dl, _ = get_dataloaders(batch_size=batch_size)

    model = build_model(freeze_backbone=False).to(DEVICE)
    # Stage 1 of this same version wrote this file; stage 2 continues from it.
    model.load_state_dict(torch.load(checkpoint_path(version), map_location=DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        if cosine
        else None
    )

    # Score the stage-1 checkpoint first so stage 2 only overwrites it if
    # fine-tuning actually improves on it.
    _, stage1_val_m = run_epoch(model, val_dl)
    stage1_val_bal_acc = stage1_val_m["balanced_accuracy"]
    print(f"Stage 1 checkpoint val_bal_acc={stage1_val_bal_acc:.4f} (fine-tuning baseline)")

    return run_training(
        model,
        train_dl,
        val_dl,
        optimizer,
        num_epochs,
        "fine_tune",
        lr,
        batch_size,
        version,
        seed,
        best_val_bal_acc=stage1_val_bal_acc,
        scheduler=scheduler,
        patience=patience,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="name of the new model version, e.g. v2 -> ml/models/efficientnet_b0_ham10000_v2.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    # Fine-tuning (stage 2) settings; defaults = v2. v1's settings were
    # --ft-lr 1e-5 --ft-epochs 10 --patience 0 --no-cosine
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument(
        "--patience", type=int, default=6, help="early stopping patience, 0 = off"
    )
    parser.add_argument("--no-cosine", action="store_true", help="constant LR in stage 2")
    args = parser.parse_args()

    train(args.version, seed=args.seed)
    fine_tune(
        args.version,
        seed=args.seed,
        num_epochs=args.ft_epochs,
        lr=args.ft_lr,
        cosine=not args.no_cosine,
        patience=args.patience or None,
    )
