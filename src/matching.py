"""Detector matching functions for FW-LNSA.

Weighted SMC is the main operational matching variant in this repository.
Standard Hamming is the classical NSA baseline, Weighted Hamming is an ablation,
and Jaccard is an optional diagnostic for active-bit overlap.
"""

from __future__ import annotations

import numpy as np


def as_binary_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    """Convert input to a two-dimensional int8 binary matrix."""

    array = np.asarray(values, dtype=np.int8)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one-dimensional or two-dimensional.")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError(f"{name} must contain only binary 0/1 values.")
    return array


def normalize_weights(weights: np.ndarray, vector_length: int) -> np.ndarray:
    """Validate and normalize feature weights for weighted matching."""

    weights_array = np.asarray(weights, dtype=float)
    if weights_array.ndim != 1:
        raise ValueError("weights must be one-dimensional.")
    if len(weights_array) != vector_length:
        raise ValueError(f"Expected {vector_length} weights, got {len(weights_array)}.")
    if np.any(weights_array < 0):
        raise ValueError("weights cannot contain negative values.")

    total = float(weights_array.sum())
    if total == 0.0:
        return np.ones(vector_length, dtype=float) / vector_length
    return weights_array / total


def hamming_distance_matrix(records: np.ndarray, detectors: np.ndarray) -> np.ndarray:
    """Return Hamming distances with shape (n_records, n_detectors)."""

    records_bin = as_binary_matrix(records, name="records")
    detectors_bin = as_binary_matrix(detectors, name="detectors")
    if records_bin.shape[1] != detectors_bin.shape[1]:
        raise ValueError("records and detectors must have the same vector length.")
    return np.sum(records_bin[:, None, :] != detectors_bin[None, :, :], axis=2)


def weighted_hamming_distance_matrix(
    records: np.ndarray,
    detectors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return Weighted Hamming distances with shape (n_records, n_detectors)."""

    records_bin = as_binary_matrix(records, name="records")
    detectors_bin = as_binary_matrix(detectors, name="detectors")
    if records_bin.shape[1] != detectors_bin.shape[1]:
        raise ValueError("records and detectors must have the same vector length.")

    normalized_weights = normalize_weights(weights, records_bin.shape[1])
    differences = records_bin[:, None, :] != detectors_bin[None, :, :]
    return np.tensordot(differences.astype(float), normalized_weights, axes=([2], [0]))


def weighted_smc_similarity_matrix(
    records: np.ndarray,
    detectors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return Weighted SMC similarities with shape (n_records, n_detectors).

    Weighted SMC sums the weights of matching bit positions. Because weights
    are normalized, similarity is in the range 0 to 1.
    """

    records_bin = as_binary_matrix(records, name="records")
    detectors_bin = as_binary_matrix(detectors, name="detectors")
    if records_bin.shape[1] != detectors_bin.shape[1]:
        raise ValueError("records and detectors must have the same vector length.")

    normalized_weights = normalize_weights(weights, records_bin.shape[1])
    matches = records_bin[:, None, :] == detectors_bin[None, :, :]
    return np.tensordot(matches.astype(float), normalized_weights, axes=([2], [0]))


def jaccard_similarity_matrix(records: np.ndarray, detectors: np.ndarray) -> np.ndarray:
    """Return Jaccard similarities with shape (n_records, n_detectors)."""

    records_bin = as_binary_matrix(records, name="records").astype(bool)
    detectors_bin = as_binary_matrix(detectors, name="detectors").astype(bool)
    if records_bin.shape[1] != detectors_bin.shape[1]:
        raise ValueError("records and detectors must have the same vector length.")

    intersection = np.sum(records_bin[:, None, :] & detectors_bin[None, :, :], axis=2)
    union = np.sum(records_bin[:, None, :] | detectors_bin[None, :, :], axis=2)
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)


def match_distance(distances: np.ndarray, threshold: float) -> np.ndarray:
    """Return True where a distance-based score is within threshold."""

    return np.asarray(distances) <= threshold


def match_similarity(similarities: np.ndarray, threshold: float) -> np.ndarray:
    """Return True where a similarity-based score meets threshold."""

    return np.asarray(similarities) >= threshold
