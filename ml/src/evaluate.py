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
    upsert_model_row,
)
from ml.src.train import DEVICE


def evaluate(version: str = DEFAULT_VERSION, notes: str = ""):
    # Architecture (and with it the input size) comes from the checkpoint's
    # file name, so this works for B0 and B2 versions alike. Evaluation runs
    # in full 32-bit precision even for models trained with amp: no gradients
    # are kept here, so memory isn't an issue.
    path = find_checkpoint(version)
    model, arch = load_trained_model(path, DEVICE)
    _, _, test_dl = get_dataloaders(input_size(arch), batch_size=32)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    print(f"Model version: {version} ({path.name})")
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
