"""Shared utilities for FW-LNSA experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimerResult:
    """Simple timing result used by experiment code."""

    name: str
    elapsed_seconds: float


@dataclass(frozen=True)
class PeakMemoryResult:
    """Peak resident-memory measurements for one measured operation."""

    baseline_rss_mb: float
    peak_rss_mb: float
    final_rss_mb: float
    peak_delta_mb: float


class PeakRSSMonitor:
    """Sample process RSS in a background thread during a measured operation.

    RSS is measured instead of relying on a single before/after subtraction.
    This captures transient NumPy allocations that disappear before an
    operation returns. If psutil is unavailable, all values are NaN.
    """

    def __init__(self, interval_seconds: float = 0.005) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive.")
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None
        self._baseline = float("nan")
        self._peak = float("nan")
        self._final = float("nan")

    def _sample(self) -> float:
        if self._process is None:
            return float("nan")
        try:
            return float(self._process.memory_info().rss / (1024 ** 2))
        except Exception:
            return float("nan")

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            value = self._sample()
            if np.isfinite(value):
                self._peak = value if not np.isfinite(self._peak) else max(self._peak, value)

    def __enter__(self) -> "PeakRSSMonitor":
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._baseline = self._sample()
            self._peak = self._baseline
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            self._process = None
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self._final = self._sample()
        if np.isfinite(self._final):
            self._peak = self._final if not np.isfinite(self._peak) else max(self._peak, self._final)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.05, self.interval_seconds * 4))

    @property
    def result(self) -> PeakMemoryResult:
        peak_delta = (
            max(0.0, self._peak - self._baseline)
            if np.isfinite(self._peak) and np.isfinite(self._baseline)
            else float("nan")
        )
        return PeakMemoryResult(
            baseline_rss_mb=float(self._baseline),
            peak_rss_mb=float(self._peak),
            final_rss_mb=float(self._final),
            peak_delta_mb=float(peak_delta),
        )


def set_random_seed(seed: int) -> None:
    """Set Python and NumPy seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path object."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a DataFrame to CSV after creating the parent folder."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=index)
    return output_path


@contextmanager
def timed_block(name: str) -> Iterator[callable]:
    """Time a code block and expose the result after the block exits."""

    start = time.perf_counter()
    result: TimerResult | None = None

    def get_result() -> TimerResult:
        if result is None:
            raise RuntimeError("Timer result is only available after the block exits.")
        return result

    try:
        yield get_result
    finally:
        elapsed = time.perf_counter() - start
        result = TimerResult(name=name, elapsed_seconds=elapsed)


def get_memory_usage_mb() -> float:
    """Return current process memory usage in MB when psutil is available."""

    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return float("nan")


def stable_partition_signature(
    X: pd.DataFrame,
    y: Sequence[int],
    original_labels: Sequence[object] | None = None,
) -> str:
    """Create a stable SHA-256 signature for one prepared data partition."""

    labels = np.asarray(y, dtype=np.int8)
    if len(X) != len(labels):
        raise ValueError("X and y must contain the same number of records.")

    digest = hashlib.sha256()
    digest.update(str(X.shape).encode("utf-8"))
    digest.update("\x1f".join(map(str, X.columns)).encode("utf-8"))

    for column in X.columns:
        values = np.asarray(X[column], dtype=np.float64)
        digest.update(values.tobytes(order="C"))
    digest.update(labels.tobytes(order="C"))

    if original_labels is not None:
        normalized = pd.Series(original_labels, dtype="object").fillna("").astype(str)
        if len(normalized) != len(X):
            raise ValueError("original_labels must match the partition length.")
        for value in normalized:
            encoded = value.encode("utf-8", errors="replace")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)

    return digest.hexdigest()


def stable_feature_selection_signature(
    features: Sequence[object],
    scores: Sequence[float],
    weights: Sequence[float],
) -> str:
    """Create a stable signature for selected features, scores, and weights."""

    feature_list = [str(value) for value in features]
    score_array = np.asarray(scores, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if not (len(feature_list) == len(score_array) == len(weight_array)):
        raise ValueError("Features, scores, and weights must have equal lengths.")

    digest = hashlib.sha256()
    for feature in feature_list:
        encoded = feature.encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    digest.update(score_array.tobytes(order="C"))
    digest.update(weight_array.tobytes(order="C"))
    return digest.hexdigest()


def stable_mapping_signature(value: Mapping[str, object]) -> str:
    """Create a stable signature for a JSON-compatible mapping."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
