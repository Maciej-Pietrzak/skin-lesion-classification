"""Model versioning: one checkpoint file per version, plus ml/results/models.csv
listing every version's test-set results.

Each training run gets a version name (v1, v2, ...) and its own checkpoint
file, so earlier models are never overwritten and can be compared side by
side (results table in the thesis) or switched back to in the backend.
ml/MODELS.md holds the narrative description of each version; models.csv is
the machine-readable summary of the same versions.
"""

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from ml.src.model import ARCHITECTURES, build_model

ML_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ML_DIR / "models"
RESULTS_DIR = ML_DIR / "results"
MODELS_CSV_PATH = RESULTS_DIR / "models.csv"

# The version scripts default to when --version isn't given (evaluate,
# gradcam, export_onnx). Bump it once a newer version becomes the main model.
DEFAULT_VERSION = "v3"

MODELS_CSV_FIELDS = [
    "version",
    "evaluated_at",
    "checkpoint",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_mel_recall",
    "test_mel_f1",
    "preprocess",
    "notes",
]


def checkpoint_path(version: str, arch: str) -> Path:
    """Where a version's checkpoint lives. The architecture is part of the file
    name (e.g. efficientnet_b2_ham10000_v5.pt), because a checkpoint is a raw
    state_dict: to load it, you must first build the matching network, and the
    file name is how evaluate/gradcam/the backend know which one to build.
    """
    return MODELS_DIR / f"{arch}_ham10000_{version}.pt"


def existing_checkpoints(version: str) -> list[Path]:
    return sorted(MODELS_DIR.glob(f"*_ham10000_{version}.pt"))


def find_checkpoint(version: str) -> Path:
    """The checkpoint file of an already trained version, whatever its arch."""
    matches = existing_checkpoints(version)
    if not matches:
        raise FileNotFoundError(f"No checkpoint for version {version} in {MODELS_DIR}")
    if len(matches) > 1:
        raise RuntimeError(f"Several checkpoints for version {version}: {matches}")
    return matches[0]


def arch_from_checkpoint(path: Path | str) -> str:
    arch = Path(path).stem.rsplit("_ham10000_", 1)[0]
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"Can't tell the architecture of {Path(path).name}: expected "
            f"<arch>_ham10000_<version>.pt with arch in {list(ARCHITECTURES)}"
        )
    return arch


def meta_path(ckpt_path: Path | str) -> Path:
    """Small JSON file next to a checkpoint (same name, .json) with settings
    that the file name can't carry but loading code must know -- currently
    how images have to be preprocessed. Unlike the .pt files it is tracked
    in git."""
    return Path(ckpt_path).with_suffix(".json")


def save_checkpoint_meta(ckpt_path: Path | str, meta: dict) -> None:
    with open(meta_path(ckpt_path), "w") as f:
        json.dump(meta, f, indent=2)


def _meta_value(ckpt_path: Path | str, key: str, default: str) -> str:
    path = meta_path(ckpt_path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f).get(key, default)


def preprocess_of(ckpt_path: Path | str) -> str:
    """The preprocess mode (dataloaders.PREPROCESS_MODES) a version was trained
    with. v1-v6 have no meta file: they all used "resize"."""
    return _meta_value(ckpt_path, "preprocess", "resize")


def augment_of(ckpt_path: Path | str) -> str:
    """The train augmentation (dataloaders.AUGMENT_MODES) of a version; v1-v7
    used "base" (v7's meta file predates the key)."""
    return _meta_value(ckpt_path, "augment", "base")


def load_trained_model(path: Path | str, device) -> tuple[nn.Module, str]:
    """Builds the network matching the checkpoint's architecture, loads its
    weights and switches it to eval mode. Returns (model, arch); arch gives the
    input size to preprocess images at (model.input_size(arch)).
    """
    arch = arch_from_checkpoint(path)
    model = build_model(arch).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, arch


def upsert_model_row(row: dict) -> None:
    """Adds this version's row to models.csv, or replaces it if the version is
    already listed -- so re-running evaluate on the same model updates its
    results instead of creating a duplicate row.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    if MODELS_CSV_PATH.exists():
        with open(MODELS_CSV_PATH, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["version"] != row["version"]]
    rows.append(row)

    with open(MODELS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MODELS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
