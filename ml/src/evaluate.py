"""Evaluates a trained model version on the held-out test set and records its
results in ml/results/models.csv."""

import argparse
from datetime import datetime

import torch
from sklearn.metrics import classification_report, confusion_matrix

from ml.src.dataloaders import get_dataloaders
from ml.src.dataset import CLASSES
from ml.src.metrics import compute_metrics
from ml.src.model import input_size
from ml.src.registry import (
    DEFAULT_VERSION,
    find_checkpoint,
    load_trained_model,
    preprocess_of,
    upsert_model_row,
)
from ml.src.train import DEVICE


def predict_proba(model, imgs: torch.Tensor) -> torch.Tensor:
    """Class probabilities, shape (B, num_classes).

    imgs is either (B, C, H, W) -- one view per image -- or (B, N, C, H, W)
    -- N crops per image ("native_crop" test set). Crops are flattened into
    one batch for the model and their softmax probabilities averaged per
    image (probabilities, not logits, so one over-confident crop can't
    dominate the vote).
    """
    if imgs.dim() == 4:
        return model(imgs).softmax(dim=1)
    b, n = imgs.shape[:2]
    probs = model(imgs.flatten(0, 1)).softmax(dim=1)
    return probs.view(b, n, -1).mean(dim=1)


def evaluate(version: str = DEFAULT_VERSION, notes: str = ""):
    # Architecture (and with it the input size) comes from the checkpoint's
    # file name, preprocessing from its meta file (v1-v6: "resize"), so this
    # works for every version. Evaluation runs in full 32-bit precision even
    # for models trained with amp: no gradients are kept here.
    path = find_checkpoint(version)
    model, arch = load_trained_model(path, DEVICE)
    preprocess = preprocess_of(path)
    # 16 crops per image: batch 8 = 128 crops per forward pass, about what
    # batch 32 is for the single-view versions, so it fits the 4 GB GPU.
    batch_size = 8 if preprocess == "native_crop" else 32
    _, _, test_dl = get_dataloaders(input_size(arch), batch_size=batch_size, preprocess=preprocess)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            probs = predict_proba(model, imgs.to(DEVICE))
            all_preds.extend(probs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    print(f"Model version: {version} ({path.name}, preprocess={preprocess})")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(CLASSES)
    print(confusion_matrix(all_labels, all_preds))

    m = compute_metrics(all_labels, all_preds)
    print(
        f"balanced_accuracy={m['balanced_accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
        f"accuracy={m['accuracy']:.4f} mel_recall={m['mel_recall']:.4f}"
    )

    upsert_model_row(
        {
            "version": version,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": path.name,
            "test_accuracy": f"{m['accuracy']:.4f}",
            "test_balanced_accuracy": f"{m['balanced_accuracy']:.4f}",
            "test_macro_f1": f"{m['macro_f1']:.4f}",
            "test_mel_recall": f"{m['mel_recall']:.4f}",
            "test_mel_f1": f"{m['mel_f1']:.4f}",
            "preprocess": preprocess,
            "notes": notes,
        }
    )
    print(f"Results for {version} saved to ml/results/models.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--notes",
        default="",
        help="short description of what changed in this version, e.g. 'lr 1e-4 + cosine'",
    )
    args = parser.parse_args()
    evaluate(args.version, args.notes)
