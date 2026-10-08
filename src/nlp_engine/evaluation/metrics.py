from __future__ import annotations

from typing import Iterable


def accuracy_score(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("true and predicted labels must be the same length")
    if not y_true:
        return 0.0
    matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return matches / len(y_true)
