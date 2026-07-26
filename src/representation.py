"""Feature representation utilities for detector-based matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinaryRepresentationResult:
    """Binary median-split representation and the fitted medians."""

    X_train_bin: np.ndarray
    X_test_bin: np.ndarray
    medians: pd.Series
    selected_features: list[str]


def fit_median_thresholds(X_train: pd.DataFrame) -> pd.Series:
    """Fit median thresholds on training data only."""

    if X_train.empty:
        raise ValueError("X_train is empty. Median thresholds cannot be fitted.")
    return X_train.median(numeric_only=True)


def transform_with_medians(X: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    """Apply training medians to convert features into binary vectors."""

    missing = [col for col in medians.index if col not in X.columns]
    if missing:
        raise ValueError(f"Input data is missing median columns: {missing}")

    aligned = X.loc[:, medians.index]
    binary = (aligned > medians).astype(np.int8).to_numpy()
    return binary


def binary_median_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_features: Sequence[str],
) -> BinaryRepresentationResult:
    """Create a leakage-safe binary median split for selected features."""

    features = list(selected_features)
    if not features:
        raise ValueError("selected_features cannot be empty.")

    missing_train = [feature for feature in features if feature not in X_train.columns]
    missing_test = [feature for feature in features if feature not in X_test.columns]
    if missing_train or missing_test:
        raise ValueError(
            f"Selected features missing from train or test data. train={missing_train}, test={missing_test}"
        )

    train_selected = X_train.loc[:, features]
    test_selected = X_test.loc[:, features]
    medians = fit_median_thresholds(train_selected)

    return BinaryRepresentationResult(
        X_train_bin=transform_with_medians(train_selected, medians),
        X_test_bin=transform_with_medians(test_selected, medians),
        medians=medians,
        selected_features=features,
    )


def validate_binary_matrix(matrix: np.ndarray, *, name: str = "matrix") -> None:
    """Validate that a matrix contains only 0 and 1 values."""

    values = np.unique(matrix)
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(f"{name} must contain only 0 and 1 values. Found: {values}")
