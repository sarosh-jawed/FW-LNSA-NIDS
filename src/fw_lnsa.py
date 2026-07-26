"""Main Feature-Weighted Lightweight Negative Selection Algorithm model.

FW-LNSA keeps the classical self and non-self logic of negative selection while
making feature importance part of detector matching. Weighted SMC is the main
operational method. Standard Hamming and Weighted Hamming are retained for
controlled comparisons, and Jaccard remains an optional diagnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .detector_generation import (
    DetectorPool,
    MatchingMethod,
    build_detector_pool,
    predict_with_detectors,
)
from .matching import (
    as_binary_matrix,
    hamming_distance_matrix,
    jaccard_similarity_matrix,
    normalize_weights,
    weighted_hamming_distance_matrix,
    weighted_smc_similarity_matrix,
)
from .utils import get_memory_usage_mb, set_random_seed


@dataclass(frozen=True)
class FWLNSAFitSummary:
    """Research-facing summary of one fitted detector pool."""

    method: str
    vector_length: int
    seed: int
    candidate_detectors: int
    retained_detectors: int
    rejected_detectors: int
    retention_rate: float
    self_space_records_available: int
    self_space_records_used: int
    unique_self_patterns: int
    self_threshold: float
    detection_threshold: float
    generation_time_sec: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for CSV result rows."""

        return {
            "method": self.method,
            "vector_length": self.vector_length,
            "seed": self.seed,
            "candidate_detectors": self.candidate_detectors,
            "retained_detectors": self.retained_detectors,
            "rejected_detectors": self.rejected_detectors,
            "retention_rate": self.retention_rate,
            "self_space_records_available": self.self_space_records_available,
            "self_space_records_used": self.self_space_records_used,
            "unique_self_patterns": self.unique_self_patterns,
            "self_threshold": self.self_threshold,
            "detection_threshold": self.detection_threshold,
            "generation_time_sec": self.generation_time_sec,
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "memory_delta_mb": self.memory_delta_mb,
        }


class FWLNSA:
    """Feature-weighted lightweight negative selection classifier.

    Parameters are explicit so every run can be reproduced from a saved table.
    The classifier expects binary feature vectors and labels where 0 is normal
    traffic and 1 is attack traffic.
    """

    def __init__(
        self,
        *,
        method: MatchingMethod = "weighted_smc",
        n_detectors: int = 1000,
        self_threshold: float = 0.80,
        detection_threshold: float = 0.75,
        random_seed: int = 42,
        max_self_samples: int | None = 10000,
        deduplicate_candidates: bool = False,
        prediction_chunk_size: int = 250,
    ) -> None:
        if n_detectors <= 0:
            raise ValueError("n_detectors must be positive.")
        if max_self_samples is not None and max_self_samples <= 0:
            raise ValueError("max_self_samples must be positive or null.")
        if prediction_chunk_size <= 0:
            raise ValueError("prediction_chunk_size must be positive.")

        self.method = method
        self.n_detectors = int(n_detectors)
        self.self_threshold = float(self_threshold)
        self.detection_threshold = float(detection_threshold)
        self.random_seed = int(random_seed)
        self.max_self_samples = max_self_samples
        self.deduplicate_candidates = bool(deduplicate_candidates)
        self.prediction_chunk_size = int(prediction_chunk_size)

        self.detector_pool_: DetectorPool | None = None
        self.feature_weights_: np.ndarray | None = None
        self.fit_summary_: FWLNSAFitSummary | None = None
        self.last_detection_time_sec_: float | None = None
        self.vector_length_: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Return True after a detector pool has been built."""

        return self.detector_pool_ is not None

    def _validate_labels(self, y: np.ndarray, expected_length: int) -> np.ndarray:
        labels = np.asarray(y, dtype=np.int8)
        if labels.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if len(labels) != expected_length:
            raise ValueError("X and y must contain the same number of records.")
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("y must contain only binary labels 0 and 1.")
        return labels

    def _prepare_weights(self, feature_weights: np.ndarray | None, vector_length: int) -> np.ndarray | None:
        weighted_method = self.method in {"weighted_hamming", "weighted_smc"}
        if not weighted_method:
            return None
        if feature_weights is None:
            raise ValueError(f"feature_weights are required for method '{self.method}'.")
        return normalize_weights(feature_weights, vector_length)

    def _sample_self_space(self, self_space: np.ndarray) -> np.ndarray:
        if self.max_self_samples is None or len(self_space) <= self.max_self_samples:
            return self_space

        rng = np.random.default_rng(self.random_seed)
        selected = rng.choice(len(self_space), size=self.max_self_samples, replace=False)
        return self_space[selected]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_weights: np.ndarray | None = None,
    ) -> "FWLNSA":
        """Build a detector pool from normal training records only."""

        X_binary = as_binary_matrix(X, name="X")
        labels = self._validate_labels(y, len(X_binary))
        self_space = X_binary[labels == 0]
        if len(self_space) == 0:
            raise ValueError("At least one normal training record is required.")

        set_random_seed(self.random_seed)
        self.vector_length_ = int(X_binary.shape[1])
        self.feature_weights_ = self._prepare_weights(feature_weights, self.vector_length_)
        sampled_self = self._sample_self_space(self_space)

        memory_before = get_memory_usage_mb()
        started = time.perf_counter()
        detector_pool = build_detector_pool(
            sampled_self,
            n_detectors=self.n_detectors,
            vector_length=self.vector_length_,
            method=self.method,
            self_threshold=self.self_threshold,
            seed=self.random_seed,
            weights=self.feature_weights_,
            deduplicate_candidates=self.deduplicate_candidates,
        )
        generation_time = time.perf_counter() - started
        memory_after = get_memory_usage_mb()

        self.detector_pool_ = detector_pool
        self.fit_summary_ = FWLNSAFitSummary(
            method=self.method,
            vector_length=self.vector_length_,
            seed=self.random_seed,
            candidate_detectors=detector_pool.candidate_count,
            retained_detectors=detector_pool.retained_count,
            rejected_detectors=detector_pool.rejected_count,
            retention_rate=detector_pool.retention_rate,
            self_space_records_available=int(len(self_space)),
            self_space_records_used=int(len(sampled_self)),
            unique_self_patterns=detector_pool.unique_self_patterns,
            self_threshold=self.self_threshold,
            detection_threshold=self.detection_threshold,
            generation_time_sec=float(generation_time),
            memory_before_mb=float(memory_before),
            memory_after_mb=float(memory_after),
            memory_delta_mb=float(memory_after - memory_before),
        )
        return self

    def _require_fitted(self) -> DetectorPool:
        if self.detector_pool_ is None or self.vector_length_ is None:
            raise RuntimeError("FWLNSA must be fitted before prediction.")
        return self.detector_pool_

    def _validate_prediction_matrix(self, X: np.ndarray) -> np.ndarray:
        X_binary = as_binary_matrix(X, name="X")
        self._require_fitted()
        if X_binary.shape[1] != self.vector_length_:
            raise ValueError(
                f"Expected {self.vector_length_} features, got {X_binary.shape[1]}."
            )
        return X_binary

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify records as normal or attack using retained detectors."""

        X_binary = self._validate_prediction_matrix(X)
        detector_pool = self._require_fitted()

        started = time.perf_counter()
        predictions = predict_with_detectors(
            X_binary,
            detector_pool,
            detection_threshold=self.detection_threshold,
            weights=self.feature_weights_,
            chunk_size=self.prediction_chunk_size,
        )
        self.last_detection_time_sec_ = float(time.perf_counter() - started)
        return predictions

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Return nearest-detector distances or strongest similarities.

        Lower scores indicate stronger matches for distance methods. Higher
        scores indicate stronger matches for similarity methods. These scores
        support later uncertainty analysis without changing binary predictions.
        """

        X_binary = self._validate_prediction_matrix(X)
        pool = self._require_fitted()
        detectors = pool.detectors

        if len(detectors) == 0:
            if self.method in {"hamming", "weighted_hamming"}:
                return np.full(len(X_binary), np.inf, dtype=float)
            return np.zeros(len(X_binary), dtype=float)

        scores = np.empty(len(X_binary), dtype=float)
        for start in range(0, len(X_binary), self.prediction_chunk_size):
            end = min(start + self.prediction_chunk_size, len(X_binary))
            chunk = X_binary[start:end]

            if self.method == "hamming":
                matrix = hamming_distance_matrix(chunk, detectors)
                scores[start:end] = np.min(matrix, axis=1)
            elif self.method == "weighted_hamming":
                matrix = weighted_hamming_distance_matrix(
                    chunk,
                    detectors,
                    self.feature_weights_,
                )
                scores[start:end] = np.min(matrix, axis=1)
            elif self.method == "weighted_smc":
                matrix = weighted_smc_similarity_matrix(
                    chunk,
                    detectors,
                    self.feature_weights_,
                )
                scores[start:end] = np.max(matrix, axis=1)
            elif self.method == "jaccard":
                matrix = jaccard_similarity_matrix(chunk, detectors)
                scores[start:end] = np.max(matrix, axis=1)
            else:
                raise ValueError(f"Unsupported matching method: {self.method}")

        return scores

    def get_detector_summary(self) -> dict[str, Any]:
        """Return detector, timing, and memory details for result tables."""

        if self.fit_summary_ is None:
            raise RuntimeError("FWLNSA must be fitted before requesting a summary.")

        summary = self.fit_summary_.to_dict()
        summary["detection_time_sec"] = self.last_detection_time_sec_
        if self.last_detection_time_sec_ is None:
            summary["total_time_sec"] = None
        else:
            summary["total_time_sec"] = (
                self.fit_summary_.generation_time_sec + self.last_detection_time_sec_
            )
        return summary
