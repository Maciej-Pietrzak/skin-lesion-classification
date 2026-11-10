"""Training loop for HAM10000 EfficientNet-B0 transfer learning."""

import argparse
import csv
from datetime import datetime

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
    best_val_bal_acc=0.0,
):
    """Shared epoch loop: trains, validates, checkpoints on val balanced
    accuracy improvement to this version's own checkpoint file, and logs every
    epoch to METRICS_LOG_PATH.

    best_val_bal_acc lets fine_tune() seed this with stage 1's score, so stage 2
    only overwrites the checkpoint if it actually beats the frozen-backbone model.
    """
    for epoch in range(1, num_epochs + 1):
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

        log_metrics_row(
            {
                "version": version,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "epoch": epoch,
                "num_epochs": num_epochs,
                "lr": lr,
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

    return best_val_bal_acc


def train(
    version: str,
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

    train_dl, val_dl, _ = get_dataloaders(batch_size=batch_size)
    model = build_model(freeze_backbone=freeze_backbone).to(DEVICE)

    # Only pass trainable params to the optimizer; with freeze_backbone=True
    # that's just the classifier head.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    return run_training(
        model, train_dl, val_dl, optimizer, num_epochs, "train", lr, batch_size, version
    )


def fine_tune(version: str, num_epochs: int = 10, batch_size: int = 32, lr: float = 1e-5):
    """Stage 2: unfreeze the whole network and keep training from stage 1's
    best weights, at a much lower LR than stage 1 (1e-3 -> 1e-5). The low LR
    matters: the backbone's pretrained filters are already good, so large
    updates here would wreck them rather than gently adapt them to skin lesions.
    """
    train_dl, val_dl, _ = get_dataloaders(batch_size=batch_size)

    model = build_model(freeze_backbone=False).to(DEVICE)
    # Stage 1 of this same version wrote this file; stage 2 continues from it.
    model.load_state_dict(torch.load(checkpoint_path(version), map_location=DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        best_val_bal_acc=stage1_val_bal_acc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="name of the new model version, e.g. v2 -> ml/models/efficientnet_b0_ham10000_v2.pt",
    )
    args = parser.parse_args()

    train(args.version)
    fine_tune(args.version)
