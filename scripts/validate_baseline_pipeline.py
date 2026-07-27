"""Run the baseline model and fair-comparison validation tests.

Usage:
    python scripts/validate_baseline_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_file = repo_root / "tests" / "test_baseline_pipeline.py"
    if not test_file.exists():
        raise FileNotFoundError(f"Validation test file not found: {test_file}")

    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "tests.test_baseline_pipeline"],
        cwd=repo_root,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print("Baseline pipeline validation passed.")


if __name__ == "__main__":
    main()
