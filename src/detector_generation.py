"""Candidate detector generation and negative selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .matching import (
    as_binary_matrix,
    hamming_distance_matrix,
    jaccard_similarity_matrix,
    weighted_hamming_distance_matrix,
    weighted_binary_similarity_matrix,
    weighted_smc_similarity_matrix,
)

MatchingMethod = Literal[
    "hamming",
    "weighted_hamming",
    "weighted_similarity",
    "weighted_smc",
    "jaccard",
]


@dataclass(frozen=True)
class DetectorPool:
    """Retained detector pool created after negative selection."""

    detectors: np.ndarray
    method: str
    threshold: float
    rejected_count: int
    candidate_count: int
    unique_self_patterns: int

    @property
    def retained_count(self) -> int:
        return int(len(self.detectors))

    @property
    def retention_rate(self) -> float:
        if self.candidate_count == 0:
            return 0.0
        return self.retained_count / self.candidate_count


def generate_candidate_detectors(
    n_detectors: int,
    vector_length: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Generate reproducible random binary candidate detectors."""

    if n_detectors < 0:
        raise ValueError("n_detectors cannot be negative.")
    if vector_length <= 0:
        raise ValueError("vector_length must be positive.")

    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_detectors, vector_length), dtype=np.int8)


def get_unique_binary_rows(matrix: np.ndarray) -> np.ndarray:
    """Remove duplicate binary rows while preserving a valid 2D shape."""

    matrix_bin = as_binary_matrix(matrix, name="matrix")
    if len(matrix_bin) == 0:
        return matrix_bin
    return np.unique(matrix_bin, axis=0).astype(np.int8)


def deduplicate_detectors(detectors: np.ndarray) -> np.ndarray:
    """Remove duplicate detectors after generation or selection."""

    return get_unique_binary_rows(detectors)


def _empty_detectors(vector_length: int) -> np.ndarray:
    return np.empty((0, vector_length), dtype=np.int8)


def negative_selection(
    candidates: np.ndarray,
    self_space: np.ndarray,
    *,
    method: MatchingMethod,
    threshold: float,
    weights: np.ndarray | None = None,
    chunk_size: int = 1000,
    deduplicate_self: bool = True,
) -> DetectorPool:
    """Reject candidate detectors that match normal self-space.

    Distance methods reject when distance <= threshold. Similarity methods
    reject when similarity >= threshold.
    """

    candidate_matrix = as_binary_matrix(candidates, name="candidates")
    self_matrix = as_binary_matrix(self_space, name="self_space")

    if candidate_matrix.shape[1] != self_matrix.shape[1]:
        raise ValueError("candidates and self_space must have the same vector length.")

    if deduplicate_self:
        self_matrix = get_unique_binary_rows(self_matrix)

    if len(candidate_matrix) == 0:
        return DetectorPool(
            detectors=_empty_detectors(self_matrix.shape[1]),
            method=method,
            threshold=threshold,
            rejected_count=0,
            candidate_count=0,
            unique_self_patterns=len(self_matrix),
        )

    if len(self_matrix) == 0:
        return DetectorPool(
            detectors=candidate_matrix.astype(np.int8),
            method=method,
            threshold=threshold,
            rejected_count=0,
            candidate_count=len(candidate_matrix),
            unique_self_patterns=0,
        )

    retained_chunks: list[np.ndarray] = []
    rejected_count = 0

    for start in range(0, len(candidate_matrix), chunk_size):
        end = min(start + chunk_size, len(candidate_matrix))
        candidate_chunk = candidate_matrix[start:end]

        if method == "hamming":
            scores = hamming_distance_matrix(self_matrix, candidate_chunk)
            rejected = np.any(scores <= threshold, axis=0)
        elif method == "weighted_hamming":
            if weights is None:
                raise ValueError("weights are required for weighted_hamming.")
            scores = weighted_hamming_distance_matrix(self_matrix, candidate_chunk, weights)
            rejected = np.any(scores <= threshold, axis=0)
        elif method in {"weighted_similarity", "weighted_smc"}:
            if weights is None:
                raise ValueError(f"weights are required for {method}.")
            scores = weighted_binary_similarity_matrix(self_matrix, candidate_chunk, weights)
            rejected = np.any(scores >= threshold, axis=0)
        elif method == "jaccard":
            scores = jaccard_similarity_matrix(self_matrix, candidate_chunk)
            rejected = np.any(scores >= threshold, axis=0)
        else:
            raise ValueError(f"Unsupported matching method: {method}")

        retained_chunks.append(candidate_chunk[~rejected])
        rejected_count += int(np.sum(rejected))

    if retained_chunks:
        retained = np.vstack(retained_chunks).astype(np.int8)
    else:
        retained = _empty_detectors(candidate_matrix.shape[1])

    return DetectorPool(
        detectors=retained,
        method=method,
        threshold=threshold,
        rejected_count=rejected_count,
        candidate_count=len(candidate_matrix),
        unique_self_patterns=len(self_matrix),
    )


def predict_with_detectors(
    X: np.ndarray,
    detector_pool: DetectorPool,
    *,
    detection_threshold: float,
    weights: np.ndarray | None = None,
    chunk_size: int = 250,
) -> np.ndarray:
    """Predict 1 for attack when any retained detector matches a record."""

    X_matrix = as_binary_matrix(X, name="X")
    detectors = detector_pool.detectors

    if len(detectors) == 0:
        return np.zeros(len(X_matrix), dtype=np.int8)

    predictions = np.zeros(len(X_matrix), dtype=np.int8)
    method = detector_pool.method

    for start in range(0, len(X_matrix), chunk_size):
        end = min(start + chunk_size, len(X_matrix))
        X_chunk = X_matrix[start:end]

        if method == "hamming":
            scores = hamming_distance_matrix(X_chunk, detectors)
            matched = np.any(scores <= detection_threshold, axis=1)
        elif method == "weighted_hamming":
            if weights is None:
                raise ValueError("weights are required for weighted_hamming prediction.")
            scores = weighted_hamming_distance_matrix(X_chunk, detectors, weights)
            matched = np.any(scores <= detection_threshold, axis=1)
        elif method in {"weighted_similarity", "weighted_smc"}:
            if weights is None:
                raise ValueError(f"weights are required for {method} prediction.")
            scores = weighted_binary_similarity_matrix(X_chunk, detectors, weights)
            matched = np.any(scores >= detection_threshold, axis=1)
        elif method == "jaccard":
            scores = jaccard_similarity_matrix(X_chunk, detectors)
            matched = np.any(scores >= detection_threshold, axis=1)
        else:
            raise ValueError(f"Unsupported detector pool method: {method}")

        predictions[start:end] = matched.astype(np.int8)

    return predictions


def build_detector_pool(
    self_space: np.ndarray,
    *,
    n_detectors: int,
    vector_length: int,
    method: MatchingMethod,
    self_threshold: float,
    seed: int = 42,
    weights: np.ndarray | None = None,
    deduplicate_candidates: bool = False,
) -> DetectorPool:
    """Generate candidate detectors and apply negative selection."""

    candidates = generate_candidate_detectors(
        n_detectors=n_detectors,
        vector_length=vector_length,
        seed=seed,
    )
    if deduplicate_candidates:
        candidates = deduplicate_detectors(candidates)

    return negative_selection(
        candidates,
        self_space,
        method=method,
        threshold=self_threshold,
        weights=weights,
    )
