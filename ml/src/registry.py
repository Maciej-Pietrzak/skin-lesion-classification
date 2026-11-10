"""Model versioning: one checkpoint file per version, plus ml/results/models.csv
listing every version's test-set results.

Each training run gets a version name (v1, v2, ...) and its own checkpoint
file, so earlier models are never overwritten and can be compared side by
side (results table in the thesis) or switched back to in the backend.
ml/MODELS.md holds the narrative description of each version; models.csv is
the machine-readable summary of the same versions.
"""

import csv
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ML_DIR / "models"
RESULTS_DIR = ML_DIR / "results"
MODELS_CSV_PATH = RESULTS_DIR / "models.csv"

# The version scripts default to when --version isn't given (evaluate,
# gradcam, export_onnx). Bump it once a newer version becomes the main model.
DEFAULT_VERSION = "v2"

MODELS_CSV_FIELDS = [
    "version",
    "evaluated_at",
    "checkpoint",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_mel_recall",
    "test_mel_f1",
    "notes",
]


def checkpoint_path(version: str) -> Path:
    return MODELS_DIR / f"efficientnet_b0_ham10000_{version}.pt"


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
