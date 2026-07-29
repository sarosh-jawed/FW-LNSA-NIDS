"""Feature representation utilities for detector-based matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinaryRepresentationResult:
    """Binary median-split representation and fitted training medians."""

    X_train_bin: np.ndarray
    X_test_bin: np.ndarray
    medians: pd.Series
    selected_features: list[str]
    X_validation_bin: np.ndarray | None = None


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
    *,
    X_validation: pd.DataFrame | None = None,
) -> BinaryRepresentationResult:
    """Create a leakage-safe binary median split for selected features."""

    features = list(selected_features)
    if not features:
        raise ValueError("selected_features cannot be empty.")

    partitions = {"train": X_train, "test": X_test}
    if X_validation is not None:
        partitions["validation"] = X_validation
    missing = {
        name: [feature for feature in features if feature not in frame.columns]
        for name, frame in partitions.items()
    }
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        raise ValueError(f"Selected features are missing from partitions: {missing}")

    train_selected = X_train.loc[:, features]
    test_selected = X_test.loc[:, features]
    medians = fit_median_thresholds(train_selected)
    validation_binary = None
    if X_validation is not None:
        validation_binary = transform_with_medians(X_validation.loc[:, features], medians)

    return BinaryRepresentationResult(
        X_train_bin=transform_with_medians(train_selected, medians),
        X_test_bin=transform_with_medians(test_selected, medians),
        medians=medians,
        selected_features=features,
        X_validation_bin=validation_binary,
    )


def validate_binary_matrix(matrix: np.ndarray, *, name: str = "matrix") -> None:
    """Validate that a matrix contains only 0 and 1 values."""

    values = np.unique(matrix)
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(f"{name} must contain only 0 and 1 values. Found: {values}")
