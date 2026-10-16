"""Atomic, fingerprinted, resumable job transactions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["status"])
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return frame


def environment():
    packages = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "psutil",
        "PyYAML",
        "matplotlib",
        "pytest",
        "ruff",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu,
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "threads": {
            k: os.environ.get(k, "UNSET")
            for k in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
        "accelerator": "CPU_ONLY_ALGORITHM; Colab tier not discoverable",
        "cgroup_memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if Path("/sys/fs/cgroup/memory.max").exists()
        else "UNKNOWN",
    }


class Job:
    """A successful job resumes only if identity and every output hash agree.

    Interrupted/failed jobs rerun; each nested expensive unit is its own Job.
    A changed dependency creates an explicit error, never silent reuse.
    """

    def __init__(self, root, name, identity):
        self.path = Path(root) / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest = self.path / "manifest.json"
        self.identity = signature(identity)
        self.name = name

    def valid(self):
        if not self.manifest.exists():
            return False
        state = json.loads(self.manifest.read_text())
        if state["identity"] != self.identity:
            raise ValueError(
                f"{self.name}: checkpoint identity mismatch; choose a new output directory"
            )
        if state["status"] not in ["VERIFIED_REAL_DATA", "VERIFIED_SYNTHETIC_ONLY"]:
            return False
        for rel, expected in state["outputs"].items():
            p = self.path / rel
            if not p.is_file() or sha256(p) != expected:
                raise ValueError(f"{self.name}: output hash mismatch: {rel}")
        return True

    def run(self, callback, *, synthetic=False):
        if self.valid():
            print("RESUME", self.name, flush=True)
            return
        write_json(
            self.manifest,
            {"identity": self.identity, "status": "RUNNING", "outputs": {}},
        )
        try:
            callback(self.path)
            files = {
                p.relative_to(self.path).as_posix(): sha256(p)
                for p in self.path.rglob("*")
                if p.is_file() and p != self.manifest and not p.name.endswith(".tmp")
            }
            if not files:
                raise ValueError("Job produced no outputs")
            for rel in files:
                if rel.endswith(".csv"):
                    pd.read_csv(self.path / rel)
            write_json(
                self.manifest,
                {
                    "identity": self.identity,
                    "status": "VERIFIED_SYNTHETIC_ONLY"
                    if synthetic
                    else "VERIFIED_REAL_DATA",
                    "outputs": files,
                    "environment": environment(),
                },
            )
            print("DONE", self.name, flush=True)
        except Exception as error:
            write_json(
                self.manifest,
                {
                    "identity": self.identity,
                    "status": "FAILED",
                    "outputs": {},
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def binary_matrix(values):
    a = np.asarray(values)
    if a.ndim != 2 or not np.all(np.isin(a, [0, 1])):
        raise ValueError("Expected a two-dimensional binary matrix before casting")
    return a.astype(np.int8)
