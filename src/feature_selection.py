"""Feature selection and feature-weight utilities for FW-LNSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Selected features and normalized feature weights."""

    selected_features: list[str]
    selected_scores: pd.DataFrame
    weights: np.ndarray
    all_scores: pd.DataFrame


def compute_mutual_information_scores(
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    *,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compute mutual information scores on the training data only."""

    if X_train.empty:
        raise ValueError("X_train is empty. Feature selection cannot run.")

    # One-hot encoded categorical indicators are discrete, while scaled flow
    # measurements remain continuous. Supplying this mask is both statistically
    # appropriate and considerably faster than treating every binary indicator
    # as a continuous nearest-neighbor variable.
    values = np.ascontiguousarray(X_train.to_numpy(dtype=np.float64, copy=True))
    discrete_mask = np.all(
        np.isclose(values, 0.0) | np.isclose(values, 1.0),
        axis=0,
    )
    labels = np.asarray(y_train, dtype=np.int8)
    scores = mutual_info_classif(
        values,
        labels,
        discrete_features=discrete_mask,
        random_state=random_seed,
        n_jobs=1,
    )
    result = pd.DataFrame(
        {
            "feature": X_train.columns,
            "mi_score": scores.astype(float),
            "is_discrete": discrete_mask,
        }
    )
    return result.sort_values("mi_score", ascending=False).reset_index(drop=True)


def normalize_feature_weights(scores: Sequence[float], *, fallback: str = "uniform") -> np.ndarray:
    """Normalize feature scores into a weight vector that sums to 1."""

    weights = np.asarray(scores, dtype=float)
    if weights.ndim != 1:
        raise ValueError("Feature scores must be a one-dimensional sequence.")
    if len(weights) == 0:
        raise ValueError("At least one score is required to create weights.")

    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = float(weights.sum())

    if total == 0.0:
        if fallback != "uniform":
            raise ValueError("All feature scores are zero and fallback is not uniform.")
        return np.ones(len(weights), dtype=float) / len(weights)

    return weights / total


def select_top_features(
    score_table: pd.DataFrame,
    fs_size: int,
    *,
    score_column: str = "mi_score",
) -> FeatureSelectionResult:
    """Select the top scored features and create normalized weights."""

    if fs_size <= 0:
        raise ValueError("fs_size must be positive.")
    if fs_size > len(score_table):
        raise ValueError(f"fs_size={fs_size} exceeds available features={len(score_table)}.")
    if "feature" not in score_table.columns or score_column not in score_table.columns:
        raise ValueError("score_table must contain feature and score columns.")

    ordered = score_table.sort_values(score_column, ascending=False).reset_index(drop=True)
    selected = ordered.head(fs_size).copy()
    weights = normalize_feature_weights(selected[score_column].to_numpy(dtype=float))
    selected["normalized_weight"] = weights

    return FeatureSelectionResult(
        selected_features=selected["feature"].tolist(),
        selected_scores=selected,
        weights=weights,
        all_scores=ordered,
    )


def mutual_information_feature_selection(
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    fs_size: int,
    *,
    random_seed: int = 42,
) -> FeatureSelectionResult:
    """Run mutual information feature selection and return top features."""

    all_scores = compute_mutual_information_scores(
        X_train,
        y_train,
        random_seed=random_seed,
    )
    return select_top_features(all_scores, fs_size)
