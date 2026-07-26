"""Shared utilities for FW-LNSA experiments."""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimerResult:
    """Simple timing result used by experiment code."""

    name: str
    elapsed_seconds: float


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
    """Time a code block.

    The yielded callable returns a TimerResult after the block has completed.
    This keeps timing code explicit without hiding research steps.
    """

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
