"""Run CICIDS2017 preprocessing and experiment-pipeline validation tests.

Usage:
    python scripts/validate_cicids2017_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_file = repo_root / "tests" / "test_cicids2017_pipeline.py"
    if not test_file.exists():
        raise FileNotFoundError(f"Validation test file not found: {test_file}")

    command = [sys.executable, "-m", "unittest", "tests/test_cicids2017_pipeline.py", "-v"]
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    print("CICIDS2017 pipeline validation passed.")


if __name__ == "__main__":
    main()
