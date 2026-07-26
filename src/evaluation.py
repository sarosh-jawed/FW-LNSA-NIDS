"""Evaluation metrics and attack-category analysis for FW-LNSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Complete binary IDS metrics for one prediction vector."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    tn: int
    fp: int
    fn: int
    tp: int
    predicted_normal: int
    predicted_attack: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "predicted_normal": self.predicted_normal,
            "predicted_attack": self.predicted_attack,
        }


def _as_binary_vector(values: Sequence[int], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int8)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError(f"{name} must contain only 0 and 1.")
    return array


def evaluate_binary_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> BinaryClassificationMetrics:
    """Calculate binary intrusion-detection metrics safely."""

    truth = _as_binary_vector(y_true, name="y_true")
    predictions = _as_binary_vector(y_pred, name="y_pred")
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have the same length.")

    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0

    return BinaryClassificationMetrics(
        accuracy=float(accuracy_score(truth, predictions)),
        precision=float(precision_score(truth, predictions, zero_division=0)),
        recall=float(recall_score(truth, predictions, zero_division=0)),
        f1=float(f1_score(truth, predictions, zero_division=0)),
        fpr=fpr,
        fnr=fnr,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        predicted_normal=int(np.sum(predictions == 0)),
        predicted_attack=int(np.sum(predictions == 1)),
    )


def attack_category_analysis(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    categories: Sequence[object],
    *,
    run_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Summarize detection behavior for Normal, DoS, Probe, R2L, and U2R.

    Normal rows report false alarms and false-alarm rate. Attack rows report
    detected attacks, missed attacks, and category recall.
    """

    truth = _as_binary_vector(y_true, name="y_true")
    predictions = _as_binary_vector(y_pred, name="y_pred")
    category_series = pd.Series(categories, dtype="object").astype(str)

    if not (len(truth) == len(predictions) == len(category_series)):
        raise ValueError("Labels, predictions, and categories must have the same length.")

    preferred_order = ["Normal", "DoS", "Probe", "R2L", "U2R", "Unknown"]
    present = list(dict.fromkeys(category_series.tolist()))
    ordered_categories = [name for name in preferred_order if name in present]
    ordered_categories.extend(name for name in present if name not in ordered_categories)

    rows: list[dict[str, Any]] = []
    metadata = dict(run_metadata or {})

    for category in ordered_categories:
        mask = category_series.to_numpy() == category
        category_truth = truth[mask]
        category_pred = predictions[mask]
        total = int(np.sum(mask))
        predicted_normal = int(np.sum(category_pred == 0))
        predicted_attack = int(np.sum(category_pred == 1))

        row: dict[str, Any] = {
            **metadata,
            "category": category,
            "total_records": total,
            "predicted_normal": predicted_normal,
            "predicted_attack": predicted_attack,
            "detected_attacks": 0,
            "missed_attacks": 0,
            "category_recall": np.nan,
            "false_alarms": 0,
            "false_alarm_rate": np.nan,
        }

        if category.lower() == "normal" or np.all(category_truth == 0):
            row["false_alarms"] = predicted_attack
            row["false_alarm_rate"] = predicted_attack / total if total else np.nan
        else:
            detected = int(np.sum((category_truth == 1) & (category_pred == 1)))
            missed = int(np.sum((category_truth == 1) & (category_pred == 0)))
            row["detected_attacks"] = detected
            row["missed_attacks"] = missed
            row["category_recall"] = detected / (detected + missed) if detected + missed else np.nan

        rows.append(row)

    return pd.DataFrame(rows)
