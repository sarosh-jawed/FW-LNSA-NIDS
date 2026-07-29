"""Main Feature-Weighted Lightweight Negative Selection Algorithm model.

The confirmatory protocol uses weighted binary similarity as the canonical
feature-weighted formulation. Standard Hamming remains the unweighted NSA
baseline. Weighted Hamming and the historical ``weighted_smc`` name remain
available for equivalence checks and backward compatibility.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .detector_generation import DetectorPool, MatchingMethod, build_detector_pool
from .matching import (
    as_binary_matrix,
    hamming_distance_matrix,
    jaccard_similarity_matrix,
    normalize_weights,
    weighted_binary_similarity_matrix,
    weighted_hamming_distance_matrix,
)
from .utils import PeakRSSMonitor, get_memory_usage_mb, set_random_seed


DISTANCE_METHODS = {"hamming", "weighted_hamming"}
WEIGHTED_METHODS = {"weighted_hamming", "weighted_similarity", "weighted_smc"}
SIMILARITY_METHODS = {"weighted_similarity", "weighted_smc", "jaccard"}


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
    peak_fit_rss_mb: float
    peak_fit_rss_delta_mb: float
    detector_array_bytes: int
    feature_weights_bytes: int
    serialized_model_bytes: int

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
            "peak_fit_rss_mb": self.peak_fit_rss_mb,
            "peak_fit_rss_delta_mb": self.peak_fit_rss_delta_mb,
            "detector_array_bytes": self.detector_array_bytes,
            "feature_weights_bytes": self.feature_weights_bytes,
            "serialized_model_bytes": self.serialized_model_bytes,
        }


class FWLNSA:
    """Feature-weighted lightweight negative selection classifier.

    The classifier expects binary feature vectors and labels where 0 denotes
    normal traffic and 1 denotes attack traffic.
    """

    def __init__(
        self,
        *,
        method: MatchingMethod = "weighted_similarity",
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
        self.last_prediction_peak_rss_mb_: float | None = None
        self.last_prediction_peak_rss_delta_mb_: float | None = None
        self.vector_length_: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self.detector_pool_ is not None

    @property
    def is_distance_method(self) -> bool:
        return self.method in DISTANCE_METHODS

    def _validate_labels(self, y: np.ndarray, expected_length: int) -> np.ndarray:
        labels = np.asarray(y, dtype=np.int8)
        if labels.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if len(labels) != expected_length:
            raise ValueError("X and y must contain the same number of records.")
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("y must contain only binary labels 0 and 1.")
        return labels

    def _prepare_weights(
        self,
        feature_weights: np.ndarray | None,
        vector_length: int,
    ) -> np.ndarray | None:
        if self.method not in WEIGHTED_METHODS:
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

    def _serialized_state_size(self, detector_pool: DetectorPool) -> int:
        state = {
            "method": self.method,
            "self_threshold": self.self_threshold,
            "detection_threshold": self.detection_threshold,
            "detectors": detector_pool.detectors,
            "weights": self.feature_weights_,
            "vector_length": self.vector_length_,
        }
        return int(len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)))

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
        with PeakRSSMonitor() as memory_monitor:
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
        memory_result = memory_monitor.result

        self.detector_pool_ = detector_pool
        detector_bytes = int(detector_pool.detectors.nbytes)
        weight_bytes = int(self.feature_weights_.nbytes) if self.feature_weights_ is not None else 0
        serialized_bytes = self._serialized_state_size(detector_pool)
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
            peak_fit_rss_mb=float(memory_result.peak_rss_mb),
            peak_fit_rss_delta_mb=float(memory_result.peak_delta_mb),
            detector_array_bytes=detector_bytes,
            feature_weights_bytes=weight_bytes,
            serialized_model_bytes=serialized_bytes,
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
            raise ValueError(f"Expected {self.vector_length_} features, got {X_binary.shape[1]}.")
        return X_binary

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Return nearest-detector distances or strongest similarities."""

        X_binary = self._validate_prediction_matrix(X)
        pool = self._require_fitted()
        detectors = pool.detectors

        if len(detectors) == 0:
            if self.is_distance_method:
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
                matrix = weighted_hamming_distance_matrix(chunk, detectors, self.feature_weights_)
                scores[start:end] = np.min(matrix, axis=1)
            elif self.method in {"weighted_similarity", "weighted_smc"}:
                matrix = weighted_binary_similarity_matrix(chunk, detectors, self.feature_weights_)
                scores[start:end] = np.max(matrix, axis=1)
            elif self.method == "jaccard":
                matrix = jaccard_similarity_matrix(chunk, detectors)
                scores[start:end] = np.max(matrix, axis=1)
            else:
                raise ValueError(f"Unsupported matching method: {self.method}")

        return scores

    def attack_scores(self, X: np.ndarray) -> np.ndarray:
        """Return scores where larger values consistently indicate attack."""

        scores = self.decision_scores(X)
        return -scores if self.is_distance_method else scores

    def predict_from_scores(
        self,
        scores: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Apply a fitted model's score orientation to a supplied threshold."""

        values = np.asarray(scores, dtype=float)
        if values.ndim != 1:
            raise ValueError("scores must be one-dimensional.")
        resolved = self.detection_threshold if threshold is None else float(threshold)
        if self.is_distance_method:
            return (values <= resolved).astype(np.int8)
        return (values >= resolved).astype(np.int8)

    def score_with_resources(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Compute decision scores with timing, throughput, and peak RSS."""

        X_binary = self._validate_prediction_matrix(X)
        started = time.perf_counter()
        with PeakRSSMonitor() as memory_monitor:
            scores = self.decision_scores(X_binary)
        elapsed = float(time.perf_counter() - started)
        memory_result = memory_monitor.result
        records = int(len(X_binary))
        throughput = float(records / elapsed) if elapsed > 0 else float("inf")
        latency_per_1000_ms = float((elapsed / records) * 1_000_000) if records else float("nan")
        self.last_detection_time_sec_ = elapsed
        self.last_prediction_peak_rss_mb_ = float(memory_result.peak_rss_mb)
        self.last_prediction_peak_rss_delta_mb_ = float(memory_result.peak_delta_mb)
        return scores, {
            "detection_time_sec": elapsed,
            "prediction_records": records,
            "prediction_records_per_sec": throughput,
            "prediction_latency_per_1000_ms": latency_per_1000_ms,
            "peak_prediction_rss_mb": float(memory_result.peak_rss_mb),
            "peak_prediction_rss_delta_mb": float(memory_result.peak_delta_mb),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify records as normal or attack using retained detectors."""

        scores, _resources = self.score_with_resources(X)
        return self.predict_from_scores(scores)

    def get_detector_summary(self) -> dict[str, Any]:
        """Return detector, timing, memory, and model-footprint details."""

        if self.fit_summary_ is None:
            raise RuntimeError("FWLNSA must be fitted before requesting a summary.")

        summary = self.fit_summary_.to_dict()
        summary["detection_time_sec"] = self.last_detection_time_sec_
        summary["peak_prediction_rss_mb"] = self.last_prediction_peak_rss_mb_
        summary["peak_prediction_rss_delta_mb"] = self.last_prediction_peak_rss_delta_mb_
        if self.last_detection_time_sec_ is None:
            summary["total_time_sec"] = None
        else:
            summary["total_time_sec"] = (
                self.fit_summary_.generation_time_sec + self.last_detection_time_sec_
            )
        return summary
