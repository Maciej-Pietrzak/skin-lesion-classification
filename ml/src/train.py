"""Training loop for HAM10000 EfficientNet (B0/B2) transfer learning."""

import argparse
import csv
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ml.src.dataloaders import PREPROCESS_MODES, get_dataloaders
from ml.src.dataset import CLASSES
from ml.src.metrics import compute_metrics
from ml.src.model import ARCHITECTURES, build_model, input_size
from ml.src.registry import (
    MODELS_DIR,
    RESULTS_DIR,
    checkpoint_path,
    existing_checkpoints,
    preprocess_of,
    save_checkpoint_meta,
)

# Append-only log of every epoch across every run, so results from different
# training experiments (hyperparams, architecture tweaks, etc.) can be
# compared later in pandas/Excel instead of only living in terminal scrollback.
# The version column says which model (= which checkpoint file) a row belongs to.
METRICS_LOG_PATH = RESULTS_DIR / "training_log.csv"
METRICS_FIELDS = [
    "version",
    "arch",
    "amp",
    "preprocess",
    "seed",
    "mel_weight",
    "timestamp",
    "stage",
    "epoch",
    "num_epochs",
    "lr",
    "head_lr",
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

def build_criterion(mel_weight: float = 1.0) -> nn.CrossEntropyLoss:
    """Cross-entropy where errors on true-melanoma images cost mel_weight times
    more than errors on any other class (all others weigh 1.0).

    This is NOT an imbalance correction -- the WeightedRandomSampler in
    get_dataloaders() already shows the model every class equally often, and
    weighting all classes by frequency on top of that would double-correct.
    It is a deliberate bias towards mel: a missed melanoma is the most
    dangerous mistake this tool can make, and v3 missed 64/165 of them on the
    test set (33 called nv). The weight applies by TRUE class, so it penalises
    missing a melanoma (mel -> nv), not raising a false alarm (nv -> mel).

    Default 1.0 = plain unweighted cross-entropy (v1-v3, and v5 onwards). Only
    v4 used 2.0: it raised mel recall but cost overall F1/accuracy, about what
    a lower mel decision threshold gives without retraining, so later versions
    tune mel with a threshold instead (see ml/MODELS.md, v4). val_loss uses
    the same weights, so it isn't
    comparable across different mel_weight values -- the metrics (balanced
    accuracy, F1, mel recall) are.
    """
    weights = torch.ones(len(CLASSES))
    weights[CLASSES.index("mel")] = mel_weight
    return nn.CrossEntropyLoss(weight=weights.to(DEVICE))


def run_epoch(model, dataloader, criterion, optimizer=None, scaler=None, amp=False):
    """One pass over dataloader. Trains if optimizer is given, else evaluates.

    amp = automatic mixed precision: inside torch.autocast most operations run
    in 16-bit floats instead of 32-bit, which halves the memory taken by the
    activations every layer keeps for the backward pass. That is what lets
    EfficientNet-B2 at 260x260 with batch 32 fit in this machine's 4 GB GPU
    (measured peak: 5.6 GB without amp -> 2.9 GB with it; without it the
    driver spills into system RAM and a batch takes ~6.3 s instead of ~0.56 s).
    The weights themselves stay 32-bit, so checkpoints look exactly as before.
    scaler (GradScaler) multiplies the loss before backward() so that small
    16-bit gradients don't round down to zero, then un-scales them before the
    optimizer step. With amp=False both are no-ops (v1-v4 behaviour).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in tqdm(dataloader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, enabled=amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, compute_metrics(all_labels, all_preds)


def run_training(
    model,
    train_dl,
    val_dl,
    criterion,
    optimizer,
    num_epochs,
    stage,
    run_info,
    best_val_bal_acc=0.0,
    scheduler=None,
    patience=None,
):
    """Shared epoch loop: trains, validates, checkpoints on val balanced
    accuracy improvement to this version's own checkpoint file, and logs every
    epoch to METRICS_LOG_PATH.

    run_info holds the settings that are fixed for the whole run (version,
    arch, amp, seed, mel_weight, batch_size); they go into every log row.

    best_val_bal_acc lets fine_tune() seed this with stage 1's score, so stage 2
    only overwrites the checkpoint if it actually beats the frozen-backbone model.

    scheduler (optional) is stepped once per epoch. patience (optional) is early
    stopping: training ends once val balanced accuracy hasn't improved for that
    many epochs in a row -- the model has stopped getting better, so further
    epochs would only cost time (and risk overfitting the train set).
    """
    amp = run_info["amp"]
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=amp)
    ckpt_path = checkpoint_path(run_info["version"], run_info["arch"])

    epochs_without_improvement = 0
    for epoch in range(1, num_epochs + 1):
        # LR actually used this epoch (changes each epoch under a scheduler).
        # Stage 1 has one param group (the head), so lr == head_lr there; in
        # stage 2 group 0 is the backbone and the last group the head.
        lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[-1]["lr"]
        train_loss, train_m = run_epoch(model, train_dl, criterion, optimizer, scaler, amp)
        val_loss, val_m = run_epoch(model, val_dl, criterion, amp=amp)

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
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best val_bal_acc={best_val_bal_acc:.4f}, saved checkpoint")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_metrics_row(
            {
                **run_info,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "epoch": epoch,
                "num_epochs": num_epochs,
                "lr": f"{lr:.2e}",
                "head_lr": f"{head_lr:.2e}",
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


def make_run_info(version, arch, amp, preprocess, seed, mel_weight, batch_size) -> dict:
    # amp is only used on CUDA; on a CPU-only machine it is switched off.
    return {
        "version": version,
        "arch": arch,
        "amp": amp and DEVICE.type == "cuda",
        "preprocess": preprocess,
        "seed": seed,
        "mel_weight": mel_weight,
        "batch_size": batch_size,
    }


def train(
    version: str,
    arch: str = "efficientnet_b2",
    amp: bool = True,
    preprocess: str = "resize",
    seed: int = 42,
    num_epochs: int = 15,
    freeze_backbone: bool = True,
    batch_size: int = 32,
    lr: float = 1e-3,
    mel_weight: float = 1.0,
):
    """Stage 1: train only the classifier head on top of frozen ImageNet features.

    preprocess: how images are fed to the model, see
    dataloaders.build_transforms(). It is saved next to the checkpoint
    (registry.save_checkpoint_meta), because evaluation and the backend must
    preprocess the same way the model was trained.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Every version gets its own file, and an existing one is never
    # overwritten -- a typo'd or reused version name would otherwise silently
    # destroy an earlier model. Checked across all architectures, so the same
    # version can't exist as both a B0 and a B2 file.
    if existing_checkpoints(version):
        raise FileExistsError(
            f"{existing_checkpoints(version)[0]} already exists -- pick a new "
            f"version name (e.g. the next vN) instead of overwriting an earlier model."
        )

    save_checkpoint_meta(checkpoint_path(version, arch), {"preprocess": preprocess})

    set_seed(seed)
    train_dl, val_dl, _ = get_dataloaders(
        input_size(arch), batch_size=batch_size, preprocess=preprocess
    )
    model = build_model(arch, freeze_backbone=freeze_backbone).to(DEVICE)

    # Only pass trainable params to the optimizer; with freeze_backbone=True
    # that's just the classifier head.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    return run_training(
        model,
        train_dl,
        val_dl,
        build_criterion(mel_weight),
        optimizer,
        num_epochs,
        "train",
        make_run_info(version, arch, amp, preprocess, seed, mel_weight, batch_size),
    )


def fine_tune(
    version: str,
    arch: str = "efficientnet_b2",
    amp: bool = True,
    seed: int = 42,
    num_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-5,
    head_lr: float | None = 3e-4,
    cosine: bool = True,
    patience: int | None = 6,
    mel_weight: float = 1.0,
):
    """Stage 2: unfreeze the whole network and keep training from stage 1's
    best weights, at a lower LR than stage 1, so the pretrained filters get
    adapted to skin lesions rather than wrecked by large updates.

    Discriminative learning rates (since v6): lr is for the backbone
    (model.features), head_lr for the classifier head. The backbone's ImageNet
    filters are already good and only need gentle adjustment, while the head
    is new and specific to our 7 classes, so it can move faster. v5 (B2 with a
    single lr=1e-4) peaked at fine-tune epoch 6 and then overfit quickly -- a
    sign the backbone was changing too fast. head_lr=None means one lr for the
    whole network (v2-v5: lr=1e-4 everywhere).

    v1 used lr=1e-5 for a fixed 10 epochs with no scheduler and no early
    stopping -- val score was still rising at the end, i.e. it stopped too
    early and learned too slowly. Cosine annealing lowers every group's LR
    smoothly towards 0 over num_epochs (keeping their 10:1 ratio): big steps
    early, fine adjustments at the end. Early stopping (patience) ends
    training once the val score stops improving.
    """
    # Stage 2 always uses stage 1's preprocessing, read from its meta file.
    preprocess = preprocess_of(checkpoint_path(version, arch))

    set_seed(seed)
    train_dl, val_dl, _ = get_dataloaders(
        input_size(arch), batch_size=batch_size, preprocess=preprocess
    )
    run_info = make_run_info(version, arch, amp, preprocess, seed, mel_weight, batch_size)

    model = build_model(arch, freeze_backbone=False).to(DEVICE)
    # Stage 1 of this same version wrote this file; stage 2 continues from it.
    model.load_state_dict(
        torch.load(checkpoint_path(version, arch), map_location=DEVICE)
    )

    # Group 0 = backbone, group 1 = head; run_training logs both groups' LR.
    optimizer = torch.optim.Adam(
        [
            {"params": model.features.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": head_lr or lr},
        ]
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        if cosine
        else None
    )

    # Score the stage-1 checkpoint first so stage 2 only overwrites it if
    # fine-tuning actually improves on it.
    criterion = build_criterion(mel_weight)
    _, stage1_val_m = run_epoch(model, val_dl, criterion, amp=run_info["amp"])
    stage1_val_bal_acc = stage1_val_m["balanced_accuracy"]
    print(f"Stage 1 checkpoint val_bal_acc={stage1_val_bal_acc:.4f} (fine-tuning baseline)")

    return run_training(
        model,
        train_dl,
        val_dl,
        criterion,
        optimizer,
        num_epochs,
        "fine_tune",
        run_info,
        best_val_bal_acc=stage1_val_bal_acc,
        scheduler=scheduler,
        patience=patience,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="name of the new model version, e.g. v5 -> ml/models/efficientnet_b2_ham10000_v5.pt",
    )
    # Defaults = v6 (EfficientNet-B2 at 260x260, mixed precision). v1-v4 were
    # trained with --arch efficientnet_b0 --no-amp (v4 also --mel-weight 2.0).
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default="efficientnet_b2")
    parser.add_argument(
        "--no-amp", action="store_true", help="train in full 32-bit precision (v1-v4)"
    )
    parser.add_argument(
        "--preprocess",
        choices=PREPROCESS_MODES,
        default="resize",
        help="resize = shrink whole image (v1-v6); native_crop = crops at original "
        "resolution, 16-crop test evaluation (v7+)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mel-weight",
        type=float,
        default=1.0,
        help="loss weight for true-mel images (both stages); 1.0 = unweighted, v4 used 2.0",
    )
    # Fine-tuning (stage 2) settings; defaults = v6 (discriminative LR).
    # v2-v5 used one LR for the whole network: --ft-lr 1e-4 --ft-head-lr 0.
    # v1: --ft-lr 1e-5 --ft-head-lr 0 --ft-epochs 10 --patience 0 --no-cosine
    parser.add_argument("--ft-lr", type=float, default=3e-5, help="backbone LR in stage 2")
    parser.add_argument(
        "--ft-head-lr",
        type=float,
        default=3e-4,
        help="classifier-head LR in stage 2; 0 = same as --ft-lr",
    )
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument(
        "--patience", type=int, default=6, help="early stopping patience, 0 = off"
    )
    parser.add_argument("--no-cosine", action="store_true", help="constant LR in stage 2")
    args = parser.parse_args()

    common = dict(
        arch=args.arch, amp=not args.no_amp, seed=args.seed, mel_weight=args.mel_weight
    )
    train(args.version, **common, preprocess=args.preprocess)
    fine_tune(
        args.version,
        **common,
        num_epochs=args.ft_epochs,
        lr=args.ft_lr,
        head_lr=args.ft_head_lr or None,
        cosine=not args.no_cosine,
        patience=args.patience or None,
    )
