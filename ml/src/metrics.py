"""Classification metrics shared by training (val set, every epoch) and
evaluation (test set), so both report exactly the same numbers."""

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from ml.src.dataset import CLASSES

MEL = CLASSES.index("mel")


def compute_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    """
    - balanced_accuracy: mean per-class recall -- the main metric (official
      ISIC 2018 metric). With nv at ~67% of the data, plain accuracy is
      dominated by nv; balanced accuracy weights all 7 classes equally.
    - macro_f1: all 7 classes equal too, but also penalizes false alarms.
    - accuracy: reported for completeness / comparison with other papers.
    - mel_recall / mel_f1: a missed melanoma is the costliest error this tool
      can make, so it's tracked on its own.
    """
    return {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
        "mel_recall": recall_score(labels, preds, labels=[MEL], average=None, zero_division=0)[0],
        "mel_f1": f1_score(labels, preds, labels=[MEL], average=None, zero_division=0)[0],
    }
