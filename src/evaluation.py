"""Evaluation metrics and attack-category analysis for FW-LNSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Research-grade binary IDS metrics for one prediction vector."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    specificity: float
    balanced_accuracy: float
    mcc: float
    pr_auc: float
    roc_auc: float
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
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.mcc,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
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


def _safe_auc(truth: np.ndarray, scores: np.ndarray | None, metric: str) -> float:
    if scores is None or len(np.unique(truth)) < 2:
        return float("nan")
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1 or len(score_array) != len(truth):
        raise ValueError("y_score must be one-dimensional and match y_true length.")
    if np.any(~np.isfinite(score_array)):
        finite = score_array[np.isfinite(score_array)]
        if finite.size == 0:
            return float("nan")
        lower = float(np.min(finite) - 1.0)
        upper = float(np.max(finite) + 1.0)
        score_array = np.nan_to_num(score_array, nan=lower, neginf=lower, posinf=upper)
    try:
        if metric == "pr":
            return float(average_precision_score(truth, score_array))
        return float(roc_auc_score(truth, score_array))
    except ValueError:
        return float("nan")


def evaluate_binary_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    y_score: Sequence[float] | None = None,
) -> BinaryClassificationMetrics:
    """Calculate binary intrusion-detection metrics safely.

    ``y_score`` must be attack-oriented, meaning larger values indicate a
    stronger attack prediction. AUC values are reported as NaN when no score is
    supplied or the partition contains only one class.
    """

    truth = _as_binary_vector(y_true, name="y_true")
    predictions = _as_binary_vector(y_pred, name="y_pred")
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have the same length.")

    score_array = None if y_score is None else np.asarray(y_score, dtype=float)
    if score_array is not None and (score_array.ndim != 1 or len(score_array) != len(truth)):
        raise ValueError("y_score must be one-dimensional and match y_true length.")

    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return BinaryClassificationMetrics(
        accuracy=float(accuracy_score(truth, predictions)),
        precision=float(precision_score(truth, predictions, zero_division=0)),
        recall=float(recall_score(truth, predictions, zero_division=0)),
        f1=float(f1_score(truth, predictions, zero_division=0)),
        fpr=fpr,
        fnr=fnr,
        specificity=specificity,
        balanced_accuracy=float(balanced_accuracy_score(truth, predictions)),
        mcc=float(matthews_corrcoef(truth, predictions)),
        pr_auc=_safe_auc(truth, score_array, "pr"),
        roc_auc=_safe_auc(truth, score_array, "roc"),
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
    """Summarize normal false alarms and attack-category recall."""

    truth = _as_binary_vector(y_true, name="y_true")
    predictions = _as_binary_vector(y_pred, name="y_pred")
    category_series = pd.Series(categories, dtype="object").astype(str)

    if not (len(truth) == len(predictions) == len(category_series)):
        raise ValueError("Labels, predictions, and categories must have the same length.")

    preferred_order = [
        "Normal",
        "DoS",
        "Probe",
        "R2L",
        "U2R",
        "Attack (Unspecified)",
        "Unknown",
    ]
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
